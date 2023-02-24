#!/usr/bin/env python
import argparse
import random

import torch
from models import Encoder
from pathlib import Path
from transformers import AutoTokenizer
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import AdamW
from pytorch_lightning import LightningModule
from loss import calculate_con_loss
from utils import extract_sentences_as_summary, evaluate_rouge, extract_topk_sentences
from utils import obtain_k_ids_from_scores, sent_degree_pos_neg_sentences_index
from dataset import TrainDataset, TestDataset
from torch.utils.data import DataLoader
from utils import save_json


class SS4Sum(LightningModule):
    def __init__(self, hparams):
        # save hparams as a Namespace if hparams is a dict
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)

        """Initialize model and tokenizer."""
        super().__init__()
        self.save_hyperparameters(hparams)

        self.tokenizer = AutoTokenizer.from_pretrained(hparams.pre_encoder)
        self.encoder = Encoder(hparams.pre_encoder)

        self.output_dir = Path(hparams.output_dir)
        self.metrics_save_path = Path(hparams.output_dir) / "metrics.json"
        self.hparams_save_path = Path(self.output_dir) / "hparams.pkl"
        self.metrics = dict()
        self.metrics["val"] = []
        self.val_metric = "rouge2"
        self.temperature = self.hparams.temperature

        n_observations_per_split = {
            "train": self.hparams.n_train,
            "val": self.hparams.n_val,
            "test": self.hparams.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}

    def re_init(self, hparams):
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)

        self.save_hyperparameters(hparams)

        self.tokenizer = AutoTokenizer.from_pretrained(hparams.pre_encoder)

        self.output_dir = Path(hparams.output_dir)
        self.metrics_save_path = Path(hparams.output_dir) / "metrics.json"
        self.hparams_save_path = Path(self.output_dir) / "hparams.pkl"
        self.metrics = dict()
        self.metrics["val"] = []
        self.val_metric = "rouge2"

        n_observations_per_split = {
            "train": self.hparams.n_train,
            "val": self.hparams.n_val,
            "test": self.hparams.n_test,
        }
        self.n_obs = {k: v if v >= 0 else None for k, v in n_observations_per_split.items()}

    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        effective_batch_size = self.hparams.train_batch_size * self.hparams.accumulate_grad_batches
        return (len(self.train_dataloader().dataset) / effective_batch_size) * self.hparams.max_epochs

    def configure_optimizers(self):
        """Prepare Adafactor optimizer and schedule"""
        no_decay = ['bias', 'LayerNorm.weight']
        encoder_parameters = [
            {'params': [p for n, p in self.encoder.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.hparams.weight_decay},
            {'params': [p for n, p in self.encoder.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        encoder_optimizer = AdamW(encoder_parameters, lr=self.hparams.learning_rate)

        total_steps = int(self.total_steps())
        encoder_scheduler = get_cosine_schedule_with_warmup(encoder_optimizer, num_warmup_steps=int(self.hparams.warmup_ratio * total_steps), num_training_steps=total_steps)
        encoder_scheduler = {"scheduler": encoder_scheduler, "interval": "step", "frequency": 1}
        return [encoder_optimizer], [encoder_scheduler]

    def training_step(self, batch, batch_idx):
        hiddens = self.encoder(batch["input_sents"], batch["attn_sents"])
        degree_loss = []
        salience_scores = []
        sampled_indexs = []

        pos_index = []
        neg_index = []
        start_idx = 0
        for s in batch["doc_lens"]:
            end_idx = start_idx + s
            doc_hidden = hiddens[start_idx:end_idx, :]
            salience_score = self.encoder.amplifier(doc_hidden)
            salience_scores.append(salience_score)
            pos_num = int(s * 0.4)

            sal_indexs = obtain_k_ids_from_scores(salience_score, self.hparams.ext_sents)
            sal_index_0, sal_index_1 = sal_indexs[0], sal_indexs[1]
            sampled_indexs.append(start_idx + sal_index_0)
            sampled_indexs.append(start_idx + sal_index_1)
            sal_indexs = torch.Tensor(sal_indexs).type_as(hiddens).long()

            sent_reps = hiddens[start_idx:end_idx]
            sim_scores = torch.mm(sent_reps, sent_reps.T)

            sim_scores = sim_scores / self.temperature
            degree_pos_ids, degree_neg_ids, degree_scores = sent_degree_pos_neg_sentences_index(sim_scores, pos_num)

            degree_scores = torch.softmax(degree_scores / (s - 1), 0)
            degree_loss.append(-torch.log(torch.sum(torch.index_select(degree_scores, 0, sal_indexs))))

            for p, n in zip(degree_pos_ids, degree_neg_ids):
                pos_index.append(p + start_idx)
                neg_index.append(n + start_idx)

            start_idx = end_idx

        degree_indexs = torch.Tensor(pos_index + neg_index).type_as(hiddens).long()
        pos_all = len(pos_index)
        targets = torch.Tensor([1.] * pos_all + [0.] * pos_all).type_as(hiddens)

        salience_scores = torch.cat(salience_scores)
        salience_scores = torch.index_select(salience_scores, 0, degree_indexs)
        amp_loss = torch.nn.functional.binary_cross_entropy_with_logits(salience_scores, targets)

        dc_loss = torch.stack(degree_loss).mean()
        sampled_indexs = torch.Tensor(sampled_indexs).type_as(degree_indexs)
        sampled_hiddens = torch.index_select(hiddens, 0, sampled_indexs)
        con_loss = calculate_con_loss(sampled_hiddens, self.hparams.temperature)

        loss = con_loss + amp_loss + dc_loss
        if batch_idx % self.hparams.log_every_n_steps == 0:
            self.logger.log_metrics({"amp_loss": amp_loss.item()})
            self.logger.log_metrics({"con_loss": con_loss.item()})
            self.logger.log_metrics({"degree_loss": dc_loss.item()})
            self.logger.log_metrics({"loss": loss.item()})
        return loss

    def validation_step(self, batch, batch_idx):
        summaries = extract_sentences_as_summary(self.encoder, batch, self.hparams.ext_sents, self.temperature)
        targets = batch["tgt"]
        return {"summaries": summaries, "labels": targets}

    def test_step(self, batch, batch_idx):
        summaries = extract_sentences_as_summary(self.encoder, batch, self.hparams.ext_sents, self.temperature)
        targets = batch["tgt"]
        return {"summaries": summaries, "labels": targets}

    def get_dataloader(self, type_path, batch_size, shuffle=False):
        n_obs = self.n_obs[type_path]
        if type_path == "train":
            dataset = TrainDataset(
                dataset=self.hparams.dataset,
                tokenizer=self.tokenizer,
                max_length=self.hparams.max_length,
                n_obs=n_obs,
            )
            return DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=shuffle,
                num_workers=self.hparams.num_workers,
                sampler=None,
                drop_last=True,
            )
        elif type_path == "val":
            dataset = TestDataset(
                tokenizer=self.tokenizer,
                dataset=self.hparams.dataset,
                type_path=type_path,
                max_length=self.hparams.max_length,
                n_obs=n_obs
            )
            return DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=shuffle,
                num_workers=self.hparams.num_workers,
                sampler=None,
            )
        else:
            dataset = TestDataset(
                tokenizer=self.tokenizer,
                dataset=self.hparams.dataset,
                type_path=type_path,
                max_length=self.hparams.max_length,
                n_obs=n_obs,
            )
            return DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=shuffle,
                num_workers=self.hparams.num_workers,
                sampler=None,
            )

    def train_dataloader(self):
        return self.get_dataloader("train", batch_size=self.hparams.train_batch_size, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader("val", batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self):
        return self.get_dataloader("test", batch_size=self.hparams.test_batch_size)

    def validation_epoch_end(self, outputs):
        summaries = []
        labels = []
        for x in outputs:
            for summary, label in zip(x["summaries"], x["labels"]):
                summaries.append(summary)
                labels.append(label)

        rouge = evaluate_rouge(summaries, labels)
        self.log(self.val_metric, rouge[self.val_metric], logger=False)

        val_metrics = dict()
        for k, v in rouge.items():
            val_metrics[f"val_{k}"] = v
        self.logger.log_metrics(val_metrics)
        self.metrics["val"].append(val_metrics)

    def test_epoch_end(self, outputs):
        summaries = []
        labels = []
        for x in outputs:
            for summary, label in zip(x["summaries"], x["labels"]):
                summaries.append(summary)
                labels.append(label)

        rouge = evaluate_rouge(summaries, labels)

        test_metrics = dict()
        for k, v in rouge.items():
            test_metrics[f"test_{k}"] = v
        self.logger.log_metrics(test_metrics)
        self.metrics["test"] = [test_metrics]

        # Log results
        od = Path(self.hparams.output_dir)
        results_file = od / "test_results.txt"
        generations_file = od / "test_generations.txt"

        with open(results_file, "a+") as writer:
            for key, val in test_metrics.items():
                msg = f"{key}: {val:.6f}\n"
                writer.write(msg)

        generations_file.open("w+").write("\n".join([" ".join(s) for s in summaries]))

