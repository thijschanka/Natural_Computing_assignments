import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sys
import pandas as pd


def quantization_error(data, means, labels=None):
    N, D = data.shape
    K, D = means.shape

    new_labels = np.zeros((N,), dtype=np.int32)

    if labels is not None:
        for k in range(K):
            true_mean = np.mean(data[labels == k], axis=0)
            mean_label = np.argmin(np.linalg.norm(true_mean - means, axis=1))

            new_labels[labels == k] = mean_label

    dist = np.zeros((N, K))
    weights = np.zeros((N, K))

    for z in range(N):
        dist[z] = np.linalg.norm(means - data[z], axis=1)
        if labels is None:
            weights[z, np.argmin(dist[z])] = 1
        else:
            weights[z, new_labels[z]] = 1

    return np.mean(dist * weights) * K


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
            fitness = quantization_error(data, particles[p])

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


population_size = 10

# Artificial_1
artificial_1_data = (np.random.rand(400, 2) - 0.5) * 2
artificial_1_labels = np.zeros((400,), dtype=np.int32)
artificial_1_labels[artificial_1_data[:, 0] >= 0.7] = 1
artificial_1_labels[
    np.all([artificial_1_data[:, 0] <= 0.3, artificial_1_data[:, 1] >= -0.2 - artificial_1_data[:, 0]], axis=0)] = 1
artificial_1_clusters = 2

# Iris
label_dict = {
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
}
iris_df = pd.read_csv("iris.data", header=None)
iris_df[4] = iris_df[4].map(lambda x: label_dict[x])
iris_data = iris_df.to_numpy()[:, :4]
iris_labels = iris_df.to_numpy()[:, 4].astype(np.int32)
iris_clusters = 3

# %%

cluster_means, labels = k_centroids(artificial_1_data, population=population_size, n_clusters=artificial_1_clusters,
                                    iterations=100)

kmeans = KMeans(n_clusters=artificial_1_clusters, n_init=10, max_iter=100).fit(artificial_1_data)

plt.rcParams["figure.figsize"] = (15, 3)
plt.subplot(131)
for k in range(artificial_1_clusters):
    plt.scatter(artificial_1_data[artificial_1_labels == k][:, 0], artificial_1_data[artificial_1_labels == k][:, 1],
                label=f"Cluster {k}")
plt.legend()
plt.xlabel("$z_1$")
plt.ylabel("$z_2$")
plt.title("True clusters")

plt.subplot(132)
for k in range(artificial_1_clusters):
    plt.scatter(artificial_1_data[labels == k][:, 0], artificial_1_data[labels == k][:, 1],
                label=f"Cluster {k}")
plt.legend()
plt.xlabel("$z_1$")
plt.ylabel("$z_2$")
plt.title(f"PSO clusters, E={quantization_error(artificial_1_data, cluster_means, artificial_1_labels):.04f}")

plt.subplot(133)
for k in range(artificial_1_clusters):
    plt.scatter(artificial_1_data[kmeans.labels_ == k][:, 0], artificial_1_data[kmeans.labels_ == k][:, 1],
                label=f"Cluster {k}")
plt.legend()
plt.xlabel("$z_1$")
plt.ylabel("$z_2$")
plt.title(
    f"K-means clusters, E={quantization_error(artificial_1_data, kmeans.cluster_centers_, artificial_1_labels):.04f}")
plt.tight_layout()
plt.savefig("q3_a1.pdf")
plt.show()

# %%

cluster_means, labels = k_centroids(iris_data, population=population_size, n_clusters=iris_clusters,
                                    iterations=100)

kmeans = KMeans(n_clusters=iris_clusters, n_init=10, max_iter=100).fit(iris_data)

plt.rcParams["figure.figsize"] = (15, 3)
plt.subplot(131)
for k in range(iris_clusters):
    plt.scatter(iris_data[iris_labels == k][:, 0], iris_data[iris_labels == k][:, 1],
                label=f"Cluster {k}")
plt.legend()
plt.xlabel("$z_1$")
plt.ylabel("$z_2$")
plt.title("True clusters")

plt.subplot(132)
for k in range(iris_clusters):
    plt.scatter(iris_data[labels == k][:, 0], iris_data[labels == k][:, 1],
                label=f"Cluster {k}")
plt.legend()
plt.xlabel("$z_1$")
plt.ylabel("$z_2$")
plt.title(f"PSO clusters, E={quantization_error(iris_data, cluster_means, iris_labels):.04f}")

plt.subplot(133)
for k in range(iris_clusters):
    plt.scatter(iris_data[kmeans.labels_ == k][:, 0], iris_data[kmeans.labels_ == k][:, 1],
                label=f"Cluster {k}")
plt.legend()
plt.xlabel("$z_1$")
plt.ylabel("$z_2$")
plt.title(f"K-means clusters, E={quantization_error(iris_data, kmeans.cluster_centers_, iris_labels):.04f}")
plt.tight_layout()
plt.savefig("q3_iris.pdf")
plt.show()
# %%

trails = 30

a1_pso_errors = []
a1_k_means_errors = []

iris_pso_errors = []
iris_k_means_errors = []

for trail in range(trails):
    pso_means, _ = k_centroids(artificial_1_data, population=population_size, n_clusters=artificial_1_clusters,
                               iterations=100)
    k_means = KMeans(n_clusters=artificial_1_clusters, n_init=10, max_iter=100).fit(artificial_1_data).cluster_centers_

    a1_pso_errors.append(quantization_error(artificial_1_data, pso_means, labels=artificial_1_labels))
    a1_k_means_errors.append(quantization_error(artificial_1_data, k_means, labels=artificial_1_labels))

    pso_means, _ = k_centroids(iris_data, population=population_size, n_clusters=iris_clusters,
                               iterations=100)
    k_means = KMeans(n_clusters=iris_clusters, n_init=10, max_iter=100).fit(iris_data).cluster_centers_

    iris_pso_errors.append(quantization_error(iris_data, pso_means, labels=iris_labels))
    iris_k_means_errors.append(quantization_error(iris_data, k_means, labels=iris_labels))


plt.rcParams["figure.figsize"] = (15, 3)

plt.subplot(121)
plt.boxplot([a1_pso_errors, a1_k_means_errors], labels=["PSO", "K-means"], vert=0)
plt.title("Artificial dataset 1")

plt.subplot(122)
plt.boxplot([iris_pso_errors, iris_k_means_errors], labels=["PSO", "K-means"], vert=0)
plt.title("Iris dataset")

plt.tight_layout()
plt.savefig("q3_boxplots.pdf")
plt.show()





#%%
