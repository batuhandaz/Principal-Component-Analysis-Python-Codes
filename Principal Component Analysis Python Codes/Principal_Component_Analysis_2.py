import numpy as np
rng=np.random.RandomState(1)
x=np.dot(rng.rand(2,2),rng.randn(2,200)).T
print(x.shape)
import matplotlib.pyplot as plt
plt.scatter(x[:,0],x[:,1])
plt.show()
from sklearn.decomposition import PCA
pca = PCA(n_components= 2)
pca.fit(x)
print(pca.components_)
print(pca.explained_variance_)
pca =PCA(n_components=1)
pca.fit(x)
x_pca = pca.transform(x)
print(x.shape)
print(x_pca.shape )
x2=pca.inverse_transform(x_pca)
plt.scatter(x[:,0], x[:,1], alpha=0.2)
plt.scatter(x2[:,0], x2[:,1], alpha=0.8)
plt.show()
