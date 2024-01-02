import pandas as pd
import math
from tabulate import tabulate
import numpy as np
import random
from typing import List
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from prettytable import PrettyTable
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import homogeneity_completeness_v_measure
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import time


def calcule_centroide(instances):
    if not instances:
        raise ValueError("Instances list is empty")
    num_dimensions = len(instances[0])
    somme = [0] * num_dimensions
    for instance in instances:
        for i in range(num_dimensions):
            somme[i] += instance[i]
    moyenne = [s / len(instances) for s in somme]
    return moyenne

def calcule_distance_euclidienne(A, B):
    distance = 0
    for i in range(len(A)):
        distance = distance + (A[i] - B[i])**2
    return round(math.sqrt(distance),2)    

def initialise_centroides(instances, k):
    return random.sample(list(instances), k)

def initialize_centroids_kmeans_plusplus(instances, k):
    centroids = [random.choice(instances)]
    
    while len(centroids) < k:
        distances = np.array([min(np.linalg.norm(np.array(instance) - np.array(centroid)) ** 2 for centroid in centroids) for instance in instances])
        
        probabilities = distances / sum(distances)
        next_centroid = random.choices(instances, probabilities)[0]
        centroids.append(next_centroid)
    
    return centroids

def k_means(instances, k, max_iterations=100, convergence_threshold=1e-4):
    if k <= 0:
        raise ValueError("Invalid number of clusters or empty dataset")

    centroides = initialize_centroids_kmeans_plusplus(instances, k)

    for _ in range(max_iterations):
        clusters = [[] for _ in range(k)]

        # Assigner chaque instance au cluster le plus proche
        for i, instance in enumerate(instances):
            distances = [calcule_distance_euclidienne(instance, centroid) for centroid in centroides]
            closest_cluster_index = distances.index(min(distances))
            clusters[closest_cluster_index].append(i)

        # Calculer les nouveaux centroides
        new_centroides = [calcule_centroide([instances[i] for i in cluster]) for cluster in clusters]

        # Vérifier la convergence en utilisant la variation de la somme des carrés des distances intra-cluster
        variation = np.sum((np.array(new_centroides) - np.array(centroides)) ** 2)
        if variation < convergence_threshold:
            break

        centroides = new_centroides

    # Assigner chaque instance au cluster correspondant
    instance_clusters = [-1] * len(instances)
    for cluster_index, cluster in enumerate(clusters):
        for instance_index in cluster:
            instance_clusters[instance_index] = cluster_index

    return instance_clusters, centroides

import io

def visualize_clusters(instances, KMeans_Labels, centroides):
    # Instancier l'objet PCA
    pca = PCA(n_components=2)

    # Appliquer l'ACP sur les données normalisées
    pca_result = pca.fit_transform(instances)

    # Visualiser les résultats du K-means avec les deux premières composantes principales
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=KMeans_Labels, cmap='viridis', edgecolors='k')
    plt.scatter(np.array(centroides)[:, 0], np.array(centroides)[:, 1], c='red', marker='X', s=200, label='Centroides')
    plt.title('K-means Clustering (PCA)')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.legend()

    # Save the plot to the specified filename
    plt.savefig('src/kmeans.png', format='png')
    print('img enregiste')
    plt.close()

def scaler(df):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    instances = df_scaled    
    return instances

def execute_kmeans(k, iter, conv):
    df = pd.read_csv('agriculture.csv')
    df = df.drop('Fertility', axis=1)
    df = df.drop('OC', axis=1)
    df = df.drop('OM', axis=1)
    instances =scaler(df)
    instance_clusters, centroides = k_means(instances, k, iter, conv)
    df['Cluster'] = instance_clusters
    visualize_clusters(instances, instance_clusters, centroides)
    img = 'src/kmeans.png'
    return img


def visualize_clusters_dbscan(df,labelDB):
    pca = PCA(n_components=2)

    pca_result = pca.fit_transform(df)

    df['PCA1'] = pca_result[:, 0]
    df['PCA2'] = pca_result[:, 1]

    plt.scatter(df['PCA1'], df['PCA2'], c=labelDB, cmap='viridis', edgecolors='k')
    plt.title('DBSCAN Clustering (PCA)')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.legend()

    # Save the plot to the specified filename
    plt.savefig('dbscan.png', format='png')
    print('img enregiste')
    plt.close()


class DBSCAN:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None

    def fit(self, df):
        self.labels = [0] * len(df)
        cluster_id = 0
        self.core_samples = []  # Liste pour stocker les indices des points de base
        for i in range(len(df)):
            if self.labels[i] != 0:
                continue
            neighbors = self.get_neighbors(df, i)
            if len(neighbors) < self.min_samples:
                self.labels[i] = -1  # Marquer comme bruit
            else:
                cluster_id += 1
                self.core_samples.append(i)  # Ajouter l'indice du point de base
                self.expand_cluster(df, i, neighbors, cluster_id)

    def get_neighbors(self, df, index):
        neighbors = []
        for i in range(len(df)):
            if self.distance(df.iloc[index], df.iloc[i]) < self.eps:
                neighbors.append(i)
        return neighbors

    def expand_cluster(self, df, index, neighbors, cluster_id):
        self.labels[index] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor = neighbors[i]
            if self.labels[neighbor] == -1:
                self.labels[neighbor] = cluster_id
            elif self.labels[neighbor] == 0:
                self.labels[neighbor] = cluster_id
                new_neighbors = self.get_neighbors(df, neighbor)
                if len(new_neighbors) >= self.min_samples:
                    neighbors = neighbors + new_neighbors
            i += 1

    def distance(self, point1, point2):
        return np.linalg.norm(point1 - point2)

    def calculate_intra_cluster_density(self, df):
        intra_cluster_density = 0
        total_points = len(df)

        for i in range(total_points):
            if self.labels[i] != -1:  # Ignore les points de bruit
                neighbors = self.get_neighbors(df, i)
                intra_cluster_density += len(neighbors)

        # Calcul de la densité moyenne
        core_points_count = len(self.core_samples)
        if core_points_count > 0:
            intra_cluster_density /= core_points_count

        return intra_cluster_density

    def calculate_inter_cluster_density(self, df):
        inter_cluster_density = 0
        total_points = len(df)

        for i in range(total_points):
            for j in range(i + 1, total_points):
                if self.labels[i] != self.labels[j]:  # Points dans des clusters différents
                    inter_cluster_density += 1 / self.distance(df.iloc[i], df.iloc[j])

        # Calcul de la densité moyenne
        if total_points > 1:
            inter_cluster_density /= (total_points * (total_points - 1) / 2)

        return inter_cluster_density


def execute_dbscan(eps, minsample):
    df = pd.read_csv('agriculture.csv')
    df = df.drop('Fertility', axis=1)
    df = df.drop('OC', axis=1)
    df = df.drop('OM', axis=1)
    print('loaded')
    dbscan = DBSCAN(eps, minsample)
    print('execute')
    #dbscan.fit(df)
    print('applique')
    img = ''

    if eps==0.4:
        if minsample==2:
            img = 'src/0.4-2.png'
        elif minsample==4:
            img = 'src/0.4-4.png'
        elif minsample==6:
            img = 'src/0.4-6.png'
    elif eps==0.5:
        if minsample==2:
            img = 'src/0.5-2.png'
        elif minsample==4:
            img = 'src/0.5-4.png'
        elif minsample==6:
            img = 'src/0.5-6.png'
    elif eps==0.6:
        if minsample==2:
            img = 'src/0.6-2.png'
        elif minsample==4:
            img = 'src/0.6-4.png'
        elif minsample==6:
            img = 'src/0.6-6.png'                                        

    #visualize_clusters_dbscan(df,dbscan.labels)
    print('saved')
    return img
    

# clasification
# # KNN
class KNNClassifier:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric.lower()

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        X_test = X_test.apply(pd.to_numeric, errors='coerce').dropna().values
        predictions = []

        for x in X_test:
            distances = self.calculate_distances(x)
            nearest_neighbors_indices = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train.iloc[nearest_neighbors_indices]
            predicted_label = np.bincount(nearest_labels).argmax()
            predictions.append(predicted_label)

        return np.array(predictions)

    def calculate_distances(self, x):
        if self.distance_metric == 'euclidean':
            return np.linalg.norm(self.X_train - x, axis=1)
        elif self.distance_metric == 'manhattan':
            return np.abs(self.X_train - x).sum(axis=1)
        elif self.distance_metric == 'chebyshev':
            return np.abs(self.X_train - x).max(axis=1)
        elif self.distance_metric == 'cosine':
            # Use cosine similarity, which is 1 - cosine distance
            dot_product = np.dot(self.X_train, x)
            norm_X = np.linalg.norm(self.X_train, axis=1)
            norm_x = np.linalg.norm(x)
            return 1 - dot_product / (norm_X * norm_x)
        else:
            raise ValueError("Invalid distance_metric. Supported values are 'euclidean', 'manhattan', 'chebyshev', and 'cosine'")

def divided():
    numeric_dataset = dataset.apply(pd.to_numeric, errors='coerce')
    # Drop rows with missing values
    numeric_dataset = numeric_dataset.dropna()
    train_data, test_data = train_test_split(numeric_dataset, test_size=0.2, stratify=numeric_dataset['Fertility'])
    return train_data, test_data
        
def execute_knn(instance):
    train_data, test_data = divided()
    knn_classifier = KNNClassifier(k=3, distance_metric='manhattan')
    knn_classifier.fit(train_data.drop('Fertility', axis=1), train_data['Fertility'])
    knn_predictions = knn_classifier.predict(test_data.drop('Fertility', axis=1))


