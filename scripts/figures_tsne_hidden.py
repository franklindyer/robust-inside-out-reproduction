import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import sklearn

model_id = "mistral-7b"
layer_num = 24
# layer_num = 32

tensors = torch.load(f"data/probe_training/{model_id}/tensors.pt")
X_mat = torch.cat(
    (tensors['correct'][layer_num], tensors['incorrect'][layer_num]), 
    0
)
y_labels = torch.tensor(
    [1]*tensors['correct'][layer_num].shape[0] + [0]*tensors['incorrect'][layer_num].shape[0]
)
X_mat = sklearn.preprocessing.StandardScaler().fit_transform(X_mat)

tsne = sklearn.manifold.TSNE(
    n_components=2, 
    learning_rate='auto',
    init='random', 
    perplexity=3
)

X_lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis(n_components=1).fit_transform(X_mat, y_labels)
print(X_lda)

lda_scores = pd.DataFrame({
    ["incorrect", "correct"][lbl]: [
        X_lda[i, 0]
        for i in range(X_lda.shape[0])
        if y_labels[i] == lbl
    ]
    for lbl in [0, 1]
})
ax = sns.histplot(
    lda_scores,
    kde=True,
    multiple="dodge",
    edgecolor="none",
    palette={"correct": "green", "incorrect": "red"}
)
# ax.get_legend().set_title(#"")
plt.title("Position along principal LDA axis")
plt.title("Separation of correct and incorrect answers by hidden rep")
plt.show()

#X_tsne = sklearn.neighbors.NeighborhoodComponentsAnalysis(
#    n_components=2, 
#    init='pca', 
#    max_iter=100, 
#    random_state=0
#).fit_transform(X_mat, y_labels)
#X_emb = tsne.fit_transform(X_tsne)

#fig, ax = plt.subplots()
#groups = pd.DataFrame(X_emb, columns=['x', 'y']).assign(category=y_labels).groupby('category')
#for name, points in groups:
#    ax.scatter(points.x, points.y, label=name, alpha=0.3)

#ax.legend()

#plt.show()
