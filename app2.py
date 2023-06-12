import numpy as np
import matplotlib.pyplot as plt

def initialize_centroids(data, k):
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    return centroids

def k_means_clustering(data, k, num_iterations):
   
    centroids = initialize_centroids(data, k)

    
    for _ in range(num_iterations):
      
        distances = np.sqrt(((data - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=0)

        
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])

      
        centroids = new_centroids

    return labels, centroids


data = np.random.randn(100, 2)


k = 3
num_iterations = 10
labels, centroids = k_means_clustering(data, k, num_iterations)

plt.scatter(data[:, 0], data[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('k-means Clustering')
plt.show()
