#!/usr/bin/env python
import torch
import pickle
import json
from pyrouge import Rouge155
import os, shutil, random, string


def obtain_k_ids_from_scores(scores, k):
    index_scores = list(enumerate(scores.tolist()))
    random.shuffle(index_scores)  # shuffle for equal score sents
    sorted_scores = sorted(index_scores, key=lambda i: i[1], reverse=True)
    extract_ids = [sorted_scores[idx][0] for idx in range(k)]
    return extract_ids


def sent_degree_pos_neg_sentences_index(sim_scores, k=2):
    degree_scores = _calculate_sentence_degree_score(sim_scores)
    sent_scores = degree_scores.tolist()
    index_scores = list(enumerate(sent_scores))
    random.shuffle(index_scores)  # shuffle for equal score sents
    sorted_scores = sorted(index_scores, key=lambda i: i[1], reverse=True)
    extract_pos_ids = [sorted_scores[idx][0] for idx in range(k)]
    extract_neg_ids = [sorted_scores[-idx - 1][0] for idx in range(k)]
    return extract_pos_ids, extract_neg_ids, degree_scores


def extract_topk_sentences(sent_scores, src_sents, k):
    extract_ids = sorted(obtain_k_ids_from_scores(sent_scores, k))
    extract_sentences = [src_sents[idx] for idx in extract_ids]
    return extract_sentences


def _calculate_sentence_degree_score(sim_scores):
    diagonal_mask = torch.eye(sim_scores.size(0)).type_as(sim_scores).bool()
    scores = torch.sum(torch.masked_fill(sim_scores, diagonal_mask, 0.), 1)
    return scores


def extract_sentences_as_summary(encoder, batch, k, temperature):
    hiddens = encoder(batch["input_sents"], batch["attn_sents"])
    sents = batch["sents"]
    summaries = []
    start_idx = 0
    for s in batch["sents_len"]:
        end_idx = start_idx + s
        if s <= k:
            summaries.append(sents[start_idx: end_idx])
            start_idx = end_idx
            continue

        sent_sim_hidden = hiddens[start_idx: end_idx, :]
        sent_sim_scores = torch.mm(sent_sim_hidden, sent_sim_hidden.T)
        sent_sim_scores = sent_sim_scores / temperature
        scores = _calculate_sentence_degree_score(sent_sim_scores)
        summaries.append(extract_topk_sentences(scores, sents[start_idx: end_idx], k))
        start_idx = end_idx
    return summaries


# copy from https://github.com/mswellhao/PacSum
def evaluate_rouge(summaries, references, remove_temp=True, rouge_args=[]):
    '''
    Args:
        summaries: [[sentence]]. Each summary is a list of strings (sentences)
        references: [[[sentence]]]. Each reference is a list of candidate summaries.
        remove_temp: bool. Whether to remove the temporary files created during evaluation.
        rouge_args: [string]. A list of arguments to pass to the ROUGE CLI.
    '''
    temp_dir = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    temp_dir = os.path.join("temp", temp_dir)
    # print(temp_dir)
    system_dir = os.path.join(temp_dir, 'system')
    model_dir = os.path.join(temp_dir, 'model')
    # directory for generated summaries
    os.makedirs(system_dir)
    # directory for reference summaries
    os.makedirs(model_dir)
    # print(temp_dir, system_dir, model_dir)

    assert len(summaries) == len(references)
    for i, (summary, candidates) in enumerate(zip(summaries, references)):
        summary_fn = '%i.txt' % i
        for j, candidate in enumerate(candidates):
            candidate_fn = '%i.%i.txt' % (i, j)
            with open(os.path.join(model_dir, candidate_fn), 'w') as f:
                #print(candidate)
                f.write('\n'.join(candidate))

        with open(os.path.join(system_dir, summary_fn), 'w') as f:
            f.write('\n'.join(summary))

    args_str = ' '.join(map(str, rouge_args))
    rouge = Rouge155(rouge_args=args_str)
    rouge.system_dir = system_dir
    rouge.model_dir = model_dir
    rouge.system_filename_pattern = '(\d+).txt'
    rouge.model_filename_pattern = '#ID#.\d+.txt'

    output = rouge.convert_and_evaluate()

    r = rouge.output_to_dict(output)
    # print(output)
    # print(r)

    # remove the created temporary files
    if remove_temp:
       shutil.rmtree(temp_dir)

    rouge_score = dict()
    rouge_score["rouge1"] = r["rouge_1_f_score"] * 100.
    rouge_score["rouge2"] = r["rouge_2_f_score"] * 100.
    rouge_score["rougeL"] = r["rouge_l_f_score"] * 100.

    return rouge_score


def pickle_save(obj, path):
    """pickle.dump(obj, path)"""
    with open(path, "wb") as f:
        return pickle.dump(obj, f)


def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, "w") as f:
        json.dump(content, f, indent=indent, **json_dump_kwargs)
