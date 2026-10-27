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

k_star_sampled_only = pd.DataFrame({
    model_id: [
        k_scores[model_id][id]["internal"]["k_star_sampled_only"]
        for id in k_scores[model_id]
    ]
    for model_id in k_scores 
})

### K-SCORE BAR PLOTS WITH PERCENT INCREASE ###

model_groups = ("mistral-7b", "llama3-8b", "gemma2-9b")

grouped_k_avg = {
    int_ext: [
        np.average([k[int_ext]["k_sampled_only"] for k in k_scores[model_id].values()])
        for model_id in model_groups
    ]
    for int_ext in ["external", "internal"] 
}
k_avg_increases = {
    "external": ["", "", ""],
    "internal": [
        f"+{(100*(grouped_k_avg['internal'][i]/grouped_k_avg['external'][i]-1)):.1f}%"
        for i in range(len(model_groups))
    ]
}
grouped_kstar_avg = {
    int_ext: [
        np.average([k[int_ext]["k_star_sampled_only"] for k in k_scores[model_id].values()])
        for model_id in model_groups
    ]
    for int_ext in ["external", "internal"] 
}
kstar_avg_increases = {
    "external": ["", "", ""],
    "internal": [
        f"+{(100*(grouped_kstar_avg['internal'][i]/grouped_kstar_avg['external'][i]-1)):.1f}%"
        for i in range(len(model_groups))
    ]
}

fig, ax = plt.subplots(nrows=2, layout='constrained')

res = ax[0].grouped_bar(
    grouped_k_avg, 
    tick_labels=model_groups, 
    group_spacing=1,
    colors=["purple", "pink"]
)
for container, name in zip(res.bar_containers, grouped_k_avg):
    ax[0].bar_label(container, labels=k_avg_increases[name], padding=3, fontsize=8)


ax[0].set_ylabel('K-score')
ax[0].set_title('Internal and external avg k-scores by model')
ax[0].legend(loc='upper left', ncols=3)
ax[0].set_ylim(0, 1)

res = ax[1].grouped_bar(
    grouped_kstar_avg, 
    tick_labels=model_groups, 
    group_spacing=1,
    colors=["purple", "pink"]
)
for container, name in zip(res.bar_containers, grouped_kstar_avg):
    ax[1].bar_label(container, labels=kstar_avg_increases[name], padding=3, fontsize=8)

ax[1].set_ylabel('K*-score')
ax[1].set_title('Internal and external avg k*-scores by model')
ax[1].legend(loc='upper left', ncols=3)
ax[1].set_ylim(0, 0.3)

plt.savefig("data/img/kscore_final_comparison.png")

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
