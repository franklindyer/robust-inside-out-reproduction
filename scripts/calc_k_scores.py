import json
import matplotlib.pyplot as plt
import sys
import os

sys.path.append("scripts")
from shared import *

def k_score_for_list(results):
    num_results = len(results)
    num_correct = len([
        ans_res for ans_res in results 
        if ans_res["correct"] or ans_res["source"] == "gold"
    ])
    
    if num_results == num_correct:
        # All sampled answers must have been correct
        return 1
    
    k_cum_sum = 0
    num_cum_incorrect = 0
    for r in results:
        if r["correct"]:
            k_cum_sum += num_cum_incorrect
        else:
            num_cum_incorrect += 1
    k_score = 1 - k_cum_sum / (num_correct * (num_results - num_correct))
    return k_score

def k_scores_for_tf(q, ans_results, key_fn):
    ans_results.sort(key=lambda r: -key_fn(r))

    # K-scores with all answer types included    
    k = k_score_for_list(ans_results)
    k_star = 1 if k == 1 else 0

    # Sources used for the main experiment
    main_sources = ["gold", "sampled", "greedy"]
    k_sampled_only = k_score_for_list([r for r in ans_results if r["source"] in main_sources])
    k_star_sampled_only = 1 if k_sampled_only == 1 else 0

    # K-scores when scored against alternate gold answers
    k_other = k_score_for_list([r for r in ans_results if r["source"] in ["gold", "other_gold"]])
    k_star_other = 1 if k_other == 1 else 0 
    
    return {
        "k": k,
        "k_star": k_star,
        "k_sampled_only": k_sampled_only,
        "k_star_sampled_only": k_star_sampled_only,
        "k_other": k_other,
        "k_star_other": k_star_other
    }

QS = load_all_questions()

for model_id in ["mistral-7b", "llama3-8b", "gemma2-9b"]:
    print(f"Calculating k-scores for {model_id}...")

    out_dir = f"data/k_scores/{model_id}"
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    ANS_OPTS = json.loads(open(f"data/answer_options/{model_id}/all.json").read())
    PROBE_SCORES = json.loads(open(f"data/probe_scores/{model_id}/all.json").read())

    tf_dir = f"data/tf_answers/{model_id}"
    for fname in os.listdir(tf_dir):
        if fname[0] == ".":
            continue
        tf_scores = json.loads(open(f"{tf_dir}/{fname}").read())
        for qid in tf_scores:
            for i in range(len(tf_scores[qid])):
                ANS_OPTS[qid][i]["ab_probs"] = tf_scores[qid][i]
                ANS_OPTS[qid][i]["probe_score"] = PROBE_SCORES[qid][i]

    ext_key = lambda r: r["ab_probs"]["original"]["a_prob"] or 0
    int_key = lambda r: r["probe_score"]

    k_scores = {
        int(sid): {
            "external": k_scores_for_tf(QS[int(sid)], ANS_OPTS[sid], ext_key),
            "internal": k_scores_for_tf(QS[int(sid)], ANS_OPTS[sid], int_key)
        }
        for sid in ANS_OPTS
        if QS[int(sid)]["split"] == "test"
    }

    with open(f"{out_dir}/all.json", 'w') as f:
        f.write(json.dumps(k_scores, indent=4, ensure_ascii=False))
