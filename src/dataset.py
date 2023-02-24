#!/usr/bin/env python
import h5py
import json
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(
        self,
        dataset,
        tokenizer,
        max_length,
        n_obs=None,
    ):
        if dataset == "cd":
            self.source_data = h5py.File("data/CNN_DM/cd.train.h5df", "r")['dataset']
        if dataset == "nyt":
            self.source_data = h5py.File("data/NYT/nyt.train.h5df", "r")['dataset']
        if dataset == "cdnyt":
            self.source_data = h5py.File("data/train.h5df", "r")['dataset']
        self.dataset = dataset
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.n_obs = n_obs

    def __len__(self):
        if self.n_obs is None:
            return len(self.source_data)
        else:
            return self.n_obs

    def __getitem__(self, index):
        data = json.loads(self.source_data[index])
        if self.dataset == "cdnyt":
            return data
        else:
            return data["article"]

    def collate_fn(self, batch):
        sents = []
        doc_lens = []

        for x in batch:
            doc = x[:12]
            doc_len = len(doc)
            if doc_len <= 5:
                continue
            doc_lens.append(doc_len)
            sents.extend(doc)

        batch_encoding = dict()
        src_sents = self.tokenizer(
            sents,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
            truncation=True
        )

        batch_encoding["input_sents"] = src_sents.input_ids
        batch_encoding["attn_sents"] = src_sents.attention_mask
        batch_encoding["doc_lens"] = doc_lens

        return batch_encoding


class TestDataset(Dataset):

    def __init__(
        self,
        type_path,
        tokenizer,
        dataset,
        max_length,
        n_obs=None,
    ):
        super().__init__()
        if type_path == "val":
            if dataset == "cd":
                self.source_data = h5py.File("data/CNN_DM/cd.validation.h5df", "r")['dataset']
            if dataset == "nyt":
                self.source_data = h5py.File("data/NYT/nyt.validation.h5df", "r")['dataset']
            if dataset == "cdnyt":
                self.source_data = h5py.File("data/val.h5df", "r")['dataset']
        else:
            if dataset == "cd":
                self.source_data = h5py.File("data/CNN_DM/cd.test.h5df", "r")['dataset']
            if dataset == "nyt":
                self.source_data = h5py.File("data/NYT/nyt.test.h5df", "r")['dataset']
            if dataset == "cdnyt":
                self.source_data = list(h5py.File("data/CNN_DM/cd.test.h5df", "r")['dataset']) + list(h5py.File("data/NYT/nyt.test.h5df", "r")['dataset'])
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.n_obs = n_obs

    def __len__(self):
        if self.n_obs is None:
            return len(self.source_data)
        else:
            return self.n_obs

    def __getitem__(self, index):
        data = json.loads(self.source_data[index])
        return data

    def collate_fn(self, batch):
        sents = []
        sents_len = []
        tgt = []

        for x in batch:
            doc = x["article"]
            sents.extend(doc)
            sents_len.append(len(doc))
            tgt.append([x["abstract"]])

        src_sents = self.tokenizer(
            sents,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )

        batch_encoding = dict()
        batch_encoding["input_sents"] = src_sents.input_ids
        batch_encoding["attn_sents"] = src_sents.attention_mask

        batch_encoding["sents"] = sents
        batch_encoding["tgt"] = tgt
        batch_encoding["sents_len"] = sents_len
        return batch_encoding
