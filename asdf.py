import torch
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from utils.kmeans import KMeans

x1 = torch.randn(100, 768) * 5 + torch.randn(1, 768)
x2 = torch.randn(100, 768) * 5 + torch.randn(1, 768)
x3 = torch.randn(100, 768) * 5 + torch.randn(1, 768)
x4 = torch.randn(100, 768) * 5 + torch.randn(1, 768)

x = torch.cat([x1, x2, x3, x4], dim=0)

kmeans = KMeans(n_clusters=4, max_iter=100, batchsize=1024, mode='cosine', init='kmeans++', seed=None)
prediction = kmeans.fit_predict(x).clone()
tsne = TSNE(n_components=2, perplexity=30, n_iter=300).fit_transform(x)

for i in range(4):
    plt.scatter(tsne[prediction == i, 0], tsne[prediction == i, 1], label=f"Cluster {i}")
plt.legend()
plt.savefig("tsne.png")