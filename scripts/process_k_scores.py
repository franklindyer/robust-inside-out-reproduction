import json
import numpy as np
import scipy

def load_k_scores(model_id):
    j = json.loads(open(f"data/k_scores/{model_id}/all.json").read())
    return j

model_ids = ["mistral-7b", "llama3-8b", "gemma2-9b"]

k_scores = {
    model_id: load_k_scores(model_id)
    for model_id in model_ids 
}

gold_sampled_ids = {
    model_id: [
        int(sid)
        for sid in open(f"data/probe_training/{model_id}/gold_sampled.txt")
        if len(sid.strip()) > 0
    ]
    for model_id in model_ids
}

correct_sampled_ids = { model_id: [] for model_id in model_ids }
for model_id in model_ids:
    ans_opts = json.loads(open(f"data/answer_options/{model_id}/all.json").read())
    for sid in ans_opts:
        if any(ans_opt["correct"] and ans_opt["source"] != "gold" for ans_opt in ans_opts[sid]):
            correct_sampled_ids[model_id].append(int(sid))

k_avgs = {
    f"{model_id}|{k_variant}|{int_ext}": \
        np.average([k[int_ext][k_variant] for k in k_scores[model_id].values()])
    for k_variant in ["k_sampled_only", "k_star_sampled_only", "k_other", "k_star_other"]
    for int_ext in ["external", "internal"]
    for model_id in model_ids
}

k_tests = {
    f"{model_id}|{k_variant}": \
        scipy.stats.ttest_ind(
            [k["external"][k_variant] for k in k_scores[model_id].values()],
            [k["internal"][k_variant] for k in k_scores[model_id].values()],
            alternative="less",
            equal_var=False
        ).pvalue
    for k_variant in ["k_sampled_only", "k_star_sampled_only"]
    for model_id in model_ids
}

k_star_avgs_unsampled = {
    f"{model_id}|{int_ext}": \
        np.average([k[int_ext]["k_star_sampled_only"] for (qid, k) in k_scores[model_id].items() \
            if not (int(qid) in gold_sampled_ids[model_id] or int(qid) in correct_sampled_ids[model_id]) ])
    for int_ext in ["external", "internal"]
    for model_id in model_ids
}

with open(f"data/analysis_data/k_tests.json", 'w') as f:
    f.write(json.dumps({
        "avgs": k_avgs,
        "tests": k_tests,
        "k_star_unsampled": k_star_avgs_unsampled
    }, indent=4, ensure_ascii=False))
