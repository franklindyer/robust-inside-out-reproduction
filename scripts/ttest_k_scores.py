import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import sys

sys.path.append("scripts")
from shared import *

def load_k_scores(model_id):
    j = json.loads(open(f"data/k_scores/{model_id}/all.json").read())
    return j

QS = load_all_questions()

k_scores = {
    model_id: load_k_scores(model_id)
    for model_id in ["mistral-7b", "llama3-8b", "gemma2-9b"]
}

k_avgs_ext = {
    model_id: {
        "k": np.average([k["external"]["k_sampled_only"] for k in k_scores[model_id].values()]),
        "k_star": np.average([k["external"]["k_star_sampled_only"] for k in k_scores[model_id].values()]),
        "k_other": np.average([k["external"]["k_other"] for k in k_scores[model_id].values()]),
        "k_other_star": np.average([k["external"]["k_star_other"] for k in k_scores[model_id].values()]),
    }
    for model_id in k_scores
}

k_avgs_int = {
    model_id: {
        "k": np.average([k["internal"]["k_sampled_only"] for k in k_scores[model_id].values()]),
        "k_star": np.average([k["internal"]["k_star_sampled_only"] for k in k_scores[model_id].values()]),
        "k_other": np.average([k["internal"]["k_other"] for k in k_scores[model_id].values()]),
        "k_other_star": np.average([k["internal"]["k_star_other"] for k in k_scores[model_id].values()]),
    }
    for model_id in k_scores
}

print(k_avgs_ext)
print(k_avgs_int)

k_results = {
    model_id: scipy.stats.ttest_ind(
        [k["external"]["k_sampled_only"] for k in k_scores[model_id].values()],
        [k["internal"]["k_sampled_only"] for k in k_scores[model_id].values()],
        alternative="less",
        equal_var=False
    )
    for model_id in k_scores
}

k_star_results = {
    model_id: scipy.stats.ttest_ind(
        [k["external"]["k_star_sampled_only"] for k in k_scores[model_id].values()],
        [k["internal"]["k_star_sampled_only"] for k in k_scores[model_id].values()],
        alternative="less",
        equal_var=False
    )
    for model_id in k_scores
}

print(k_results)
print(k_star_results)

model_id = "mistral-7b"

k_decreased_ids = [
    int(qid)
    for qid in k_scores[model_id]
    if k_scores[model_id][qid]["external"]["k_other"] \
        > k_scores[model_id][qid]["internal"]["k_other"]
]

k_by_diffs = [
    (
        qid,
        k_scores[model_id][qid]["external"]["k_sampled_only"] - k_scores[model_id][qid]["internal"]["k_sampled_only"]
    )
    for qid in k_scores[model_id]
]
k_by_diffs.sort(key=lambda x: x[1])

print(k_by_diffs[:20])
print(k_by_diffs[-20:])

gold_sampled_ids = [
    int(sid.strip())
    for sid in open(f"data/probe_training/{model_id}/gold_sampled.txt")
    if len(sid) > 1
]

k_dec_gold_sampled = [id for id in k_decreased_ids if id in gold_sampled_ids]

print(k_decreased_ids[:10])
print(gold_sampled_ids[:10])
print(k_dec_gold_sampled[:20])
print(len(k_decreased_ids), len(gold_sampled_ids), len(k_dec_gold_sampled))
# print(len([id for id in k_dec_gold_sampled if QS[id]["split"] == "train"]))

dk = {
    qid: k_scores[model_id][qid]["external"]["k_sampled_only"] - k_scores[model_id][qid]["internal"]["k_sampled_only"] 
    for qid in k_scores[model_id]
}
dk_avg_sampled = np.average([dk[qid] for qid in k_scores[model_id] if int(qid) in gold_sampled_ids])
dk_avg_not_sampled = np.average([dk[qid] for qid in k_scores[model_id] if not int(qid) in gold_sampled_ids])

for model_id in ["mistral-7b", "llama3-8b", "gemma2-9b"]:
    ext_k_1 = [
        qid for qid in k_scores[model_id]
        if k_scores[model_id][qid]["external"]["k_star_sampled_only"] == 1
        and not int(qid) in gold_sampled_ids
    ]
    int_k_1 = [
        qid for qid in k_scores[model_id]
        if k_scores[model_id][qid]["internal"]["k_star_sampled_only"] == 1
        and not int(qid) in gold_sampled_ids
    ]
    total = len(list(k_scores[model_id].keys()))
    print(f"Proportion of answers both unsampled and EXT k* = 1 for {model_id}: {len(ext_k_1)/total:.2f}")
    print(f"Proportion of answers both unsampled and INT k* = 1 for {model_id}: {len(int_k_1)/total:.2f}")

print("Average k-delta when gold is sampled:", dk_avg_sampled)
print("Average k-delta when gold is not sampled:", dk_avg_not_sampled)
