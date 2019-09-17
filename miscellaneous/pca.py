import numpy as np
import cupy as cp
import sklearn.datasets as datasets
import matplotlib.pyplot as plt

data_obj = datasets.fetch_olivetti_faces()
data = data_obj.data

whiten = data - data.mean(axis=0)
whiten = cp.asarray(whiten)
cov = whiten.T @ whiten

us, vals, vs = cp.linalg.svd(cov)     # svd
# vals, us = cp.linalg.eigh(cov)        # eig

k = 10

indices = cp.argsort(vals)[::-1]
us = us[:,indices]
u = us[:,:k]

feat_face = (cov @ u).T

print(f'keep top-{k} dimensions')

feat_face = feat_face.get()

plt.imshow(feat_face[0].reshape((64,64)).astype(np.uint8), cmap='gray')
plt.show()