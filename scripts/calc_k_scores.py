import json
import matplotlib.pyplot as plt
import sys
import os

sys.path.append("scripts")
from shared import *

QS = load_all_questions()

case_study_ids = [
    1056,
    6519
]

case_studies = defaultdict(lambda: {})

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

    for qid in case_study_ids:
        case_studies[model_id][int(qid)] = {
            "question": QS[qid]["question"],
            "k_ext": k_scores[qid]["external"]["k_sampled_only"],
            "k_int": k_scores[qid]["internal"]["k_sampled_only"],
            "top10_ext": [
                {
                    "score": ext_key(r),
                    "text": r["text"],
                    "correct": r["correct"]
                } 
                for r in sorted(ANS_OPTS[str(qid)], key=lambda x: -ext_key(x))
                if r["source"] in ["gold", "sampled"]
            ][:10],
            "top10_int": [
                {
                    "score": int_key(r),
                    "text": r["text"],
                    "correct": r["correct"]
                } 
                for r in sorted(ANS_OPTS[str(qid)], key=lambda x: -int_key(x))
                if r["source"] in ["gold", "sampled"]
            ][:10]

        }
    
    with open(f"{out_dir}/all.json", 'w') as f:
        f.write(json.dumps(k_scores, indent=4, ensure_ascii=False))
    
with open(f"data/analysis_data/case_studies.json", 'w') as f:
    f.write(json.dumps(case_studies, indent=4, ensure_ascii=False))
