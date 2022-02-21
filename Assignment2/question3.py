import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sys
import pandas as pd



def k_centroids(data, population, n_clusters, w=0.5, alpha1=0.25, alpha2=0.25, iterations=100):
    N, D = data.shape

    particles = np.zeros((population, n_clusters, D))
    for p in range(population):
        particles[p] = data[np.random.choice(N, n_clusters, replace=False)]
    velocity = np.zeros((population, n_clusters, D))

    best_local_fitness = np.zeros(population)
    best_local_fitness[:] = sys.maxsize
    best_local_particle = np.zeros((population, n_clusters, D))

    global_best_fitness = sys.maxsize
    global_best_particle = np.zeros((n_clusters, D))

    for i in range(iterations):
        for p in range(population):
            dist = np.zeros((N, n_clusters))
            weights = np.zeros((N, n_clusters))

            for z in range(N):
                dist[z] = np.linalg.norm(particles[p] - data[z], axis=1)
                weights[z, np.argmin(dist[z])] = 1

            fitness = np.mean(dist * weights)

            if fitness < best_local_fitness[p]:
                best_local_particle[p] = particles[p]
                best_local_fitness[p] = fitness

        if np.min(best_local_fitness) < global_best_fitness:
            global_best_fitness = np.min(best_local_fitness)
            global_best_particle = best_local_particle[np.argmin(best_local_fitness)]

        for p in range(population):
            r1 = np.random.rand(1)
            r2 = np.random.rand(1)

            velocity[p] = w * velocity[p] \
                          + alpha1 * r1 * (best_local_particle[p] - particles[p]) \
                          + alpha2 * r2 * (global_best_particle - particles[p])

            particles[p] = particles[p] + velocity[p]

    labels = np.zeros(N)

    for z in range(N):
        dist = np.linalg.norm(global_best_particle - data[z], axis=1)
        labels[z] = np.argmin(dist)

    return global_best_particle, labels


n_clusters = 2
population_size = 15
data = np.random.rand(100, 4)

cluster_means, labels = k_centroids(data, population=population_size, n_clusters=n_clusters, iterations=100)

for k in range(n_clusters):
    plt.scatter(data[labels == k][:, 0], data[labels == k][:, 1], label=f"Cluster {k}")
plt.legend()
plt.title("PSO clustering")
plt.show()

print(f"PSO means: {cluster_means}")


kmeans = KMeans(n_clusters=n_clusters, random_state=0, max_iter=100).fit(data)

for k in range(n_clusters):
    plt.scatter(data[kmeans.labels_ == k][:, 0], data[kmeans.labels_ == k][:, 1], label=f"Cluster {k}")
plt.legend()
plt.title("K-means clustering")
plt.show()
print(f"K-means: {kmeans.cluster_centers_}")

# %%
label_dict = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
}

iris_df = pd.read_csv("iris.data", header=None)
iris_df[4] = iris_df[4].map(lambda x: label_dict[x])

iris_data = iris_df.to_numpy()[:, :4]
iris_labels = iris_df.to_numpy()[:, 4]

iris_clusters = 3

for k in range(iris_clusters):
    plt.scatter(iris_data[iris_labels == k][:, 0], iris_data[iris_labels == k][:, 1], label=f"Cluster {k}")
plt.legend()
plt.title("Actual clustering")
plt.show()


cluster_means, labels = k_centroids(iris_data, population=population_size, n_clusters=iris_clusters, iterations=100)

for k in range(iris_clusters):
    plt.scatter(iris_data[labels == k][:, 0], iris_data[labels == k][:, 1], label=f"Cluster {k}")
plt.legend()
plt.title("PSO clustering")
plt.show()

print(f"PSO means: {cluster_means}")


kmeans = KMeans(n_clusters=iris_clusters, random_state=0, max_iter=100).fit(iris_data)

for k in range(iris_clusters):
    plt.scatter(iris_data[kmeans.labels_ == k][:, 0], iris_data[kmeans.labels_ == k][:, 1], label=f"Cluster {k}")
plt.legend()
plt.title("K-means clustering")
plt.show()
print(f"K-means: {kmeans.cluster_centers_}")

#%%
