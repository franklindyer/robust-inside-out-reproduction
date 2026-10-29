import json
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import numpy as np
import pandas as pd
import seaborn as sns

### K-SCORE BAR PLOTS WITH PERCENT INCREASE ###

model_groups = ("mistral-7b", "llama3-8b", "gemma2-9b")

k_data = json.loads(open("data/analysis_data/k_tests.json").read())

for k_suffix in ["_sampled_only", "_other"]:

    grouped_k_avg = {
        int_ext: [
            k_data["avgs"][f"{model_id}|k{k_suffix}|{int_ext}"]
            for model_id in model_groups
        ]
        for int_ext in ["external", "internal"] 
    }

    k_avg_increases = {
        "external": ["", "", ""],
        "internal": [
            f"{(100*(grouped_k_avg['internal'][i]/grouped_k_avg['external'][i]-1)):+.1f}%"
            for i in range(len(model_groups))
        ]
    }

    grouped_kstar_avg = {
        int_ext: [
            k_data["avgs"][f"{model_id}|k_star{k_suffix}|{int_ext}"]
            for model_id in model_groups
        ]
        for int_ext in ["external", "internal"] 
    }

    kstar_avg_increases = {
        "external": ["", "", ""],
        "internal": [
            f"{(100*(grouped_kstar_avg['internal'][i]/grouped_kstar_avg['external'][i]-1)):+.1f}%"
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
    ax[1].set_ylim(0, 0.4)

    plt.savefig(f"data/img/k{k_suffix}_final_comparison.svg")

### PER-QUESTION DELTA K

k_scores = {
    model_id: json.loads(open(f"data/k_scores/{model_id}/all.json").read()) 
    for model_id in model_groups
}

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
    bins=np.linspace(-1, 1, 30),
    kde=True,
    multiple="dodge",
    edgecolor="none",
    hue_order=["mistral-7b", "llama3-8b", "gemma2-9b"],
    palette={'mistral-7b': 'blue', 'llama3-8b': 'orange', 'gemma2-9b': 'green'}
)
ax.get_legend().set_title("Column")
plt.title("ΔK between external and internal scoring")

plt.savefig("data/img/delta_k_distribution.svg")
plt.clf()

### QUESTIONS FOR WHICH K* INCREASES BETWEEN MODEL

k1_ext_ids = {
    model_id: set([
        qid
        for qid in k_scores[model_id]
        if k_scores[model_id][qid]["external"]["k_sampled_only"] == 1
    ])
    for model_id in model_groups
}

k1_int_ids = {
    model_id: set([
        qid
        for qid in k_scores[model_id]
        if k_scores[model_id][qid]["internal"]["k_sampled_only"] == 1
    ])
    for model_id in model_groups
}

k_jump_ids = {
    model_id: set([
        qid
        for qid in k_scores[model_id]
        if k_scores[model_id][qid]["external"]["k_sampled_only"] \
            < k_scores[model_id][qid]["internal"]["k_sampled_only"]
    ])
    for model_id in model_groups
}

k_star_jump_ids = {
    model_id: set([
        qid
        for qid in k_scores[model_id]
        if k_scores[model_id][qid]["external"]["k_star_sampled_only"] == 0
            and k_scores[model_id][qid]["internal"]["k_star_sampled_only"] == 1
    ])
    for model_id in model_groups
}

def intersect_models(ks):
    return list(ks["mistral-7b"] & ks["llama3-8b"] & ks["gemma2-9b"])

print(intersect_models(k_jump_ids)[:30])
print(intersect_models(k_star_jump_ids)[:30])

v = venn3(
    [set(k_jump_ids[model_id]) for model_id in model_groups],
    model_groups,
)

plt.savefig("data/img/k_jump_overlaps.svg")
plt.clf()

v = venn3(
    [set(k1_ext_ids[model_id]) for model_id in model_groups],
    model_groups,
)

plt.savefig("data/img/k1_ext_overlaps.svg")
plt.clf()

v = venn3(
    [set(k1_int_ids[model_id]) for model_id in model_groups],
    model_groups,
)

plt.savefig("data/img/k1_int_overlaps.svg")
plt.clf()


venn3(
    [set(k_star_jump_ids[model_id]) for model_id in model_groups],
    model_groups,
)

plt.savefig("data/img/k_star_jump_overlaps.svg")
plt.clf()
