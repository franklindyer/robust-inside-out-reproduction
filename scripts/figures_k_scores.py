import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def load_k_scores(model_id):
    j = json.loads(open(f"data/k_scores/{model_id}/all.json").read())
    return j

k_scores = {
    model_id: load_k_scores(model_id)
    for model_id in ["mistral-7b", "llama3-8b", "gemma2-9b"]
}

k_sampled_only = pd.DataFrame({
    model_id: [
        k_scores[model_id][id]["internal"]["k_sampled_only"]
        for id in k_scores[model_id]
    ]
    for model_id in k_scores 
})

k_delta = pd.DataFrame({
    model_id: [
        k_scores[model_id][id]["internal"]["k_sampled_only"]
        - k_scores[model_id][id]["external"]["k_sampled_only"]
        for id in k_scores[model_id]
    ]
    for model_id in k_scores 
})

plt.figure()

ax = sns.histplot(
    k_delta,
    kde=True,
    multiple="dodge",
    hue_order=["mistral-7b", "llama3-8b", "gemma2-9b"],
    palette={'mistral-7b': 'blue', 'llama3-8b': 'purple', 'gemma2-9b': 'red'}
)
ax.get_legend().set_title("Column")

#for model_id in k_scores:
#    sns.histplot(
#        pd.DataFrame(d_frame)[[model_id]],
#        hue="group",
#        kde=True,
#        label=model_id,
#        multiple="dodge"
#    )

plt.show()
