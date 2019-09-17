import numpy as np
import cupy as cp
import sklearn.datasets as datasets
import matplotlib.pyplot as plt

data_obj = datasets.fetch_olivetti_faces()
data = data_obj.data

whiten = data - data.mean(axis=0)
whiten = cp.asarray(whiten)
cov = whiten.T @ whiten

ss, vs = cp.linalg.eigh(cov)

indices = cp.argsort(ss)[::-1]

ss = ss[indices]
vs = vs[:,indices]

sumover = ss.sum()
k = 0
eig_feat = 0.
while eig_feat < .95*sumover:
    eig_feat += ss[k]
    k += 1

eigval = ss[:k]
eigvec = vs[:,:k]

feat_face = (cov @ eigvec).T

print(f'keep top-{k} dimensions')

feat_face = feat_face.get()

plt.imshow(feat_face[0].reshape((64,64)).astype(np.uint8), cmap='gray')
plt.show()