import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy

def load_k_scores(model_id):
    j = json.loads(open(f"data/k_scores/{model_id}/all.json").read())
    return j

k_scores = {
    model_id: load_k_scores(model_id)
    for model_id in ["mistral-7b", "llama3-8b", "gemma2-9b"]
}

k_avgs_ext = {
    model_id: {
        "k": np.average([k["external"]["k_sampled_only"] for k in k_scores[model_id].values()]),
        "k_star": np.average([k["external"]["k_star_sampled_only"] for k in k_scores[model_id].values()])
    }
    for model_id in k_scores
}

k_avgs_int = {
    model_id: {
        "k": np.average([k["internal"]["k_sampled_only"] for k in k_scores[model_id].values()]),
        "k_star": np.average([k["internal"]["k_star_sampled_only"] for k in k_scores[model_id].values()])
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
