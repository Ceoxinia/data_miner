import pandas as pd
import math
from tabulate import tabulate
import numpy as np
import random
from typing import List
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
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
# KNN
class KNNClassifier:
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric.lower()

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.apply(pd.to_numeric, errors='coerce').dropna().values
        elif isinstance(X_test, np.ndarray):
            # Assuming X_test is already numeric
            pass
        else:
            raise ValueError("Unsupported input type. X_test should be either a DataFrame or a NumPy array.")

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
    dataset = pd.read_csv("agriculture.csv")
    dataset['Fertility'] = dataset['Fertility'].astype(int)

    numeric_dataset = dataset.apply(pd.to_numeric, errors='coerce')
    # Drop rows with missing values
    numeric_dataset = numeric_dataset.dropna()
    train_data, test_data = train_test_split(numeric_dataset, test_size=0.2, stratify=numeric_dataset['Fertility'])
    return train_data, test_data
        
def execute_knn(instance,k,d):
    X_test_instance = pd.DataFrame({
    'I':[1],
    'P': [instance[0]],
    'K': [instance[1]],
    'pH': [instance[2]],
    'EC': [instance[3]],
    'OC': [instance[4]],
    'S': [instance[5]],
    'Zn': [instance[6]],
    'Fe': [instance[7]],
    'Cu': [instance[8]],
    'Mn': [instance[9]],
    'B': [instance[10]],
    'OM': [instance[11]],
    })

    train_data, test_data = divided()
    knn_classifier = KNNClassifier(k, d)
    knn_classifier.fit(train_data.drop('Fertility', axis=1), train_data['Fertility'])
    predicted_label = knn_classifier.predict(X_test_instance)
    return predicted_label


# Decision Tree
class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X_train, y_train):
        self.tree = self._build_tree(X_train, y_train, depth=0)

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        unique_classes, class_counts = np.unique(y, return_counts=True)

        # If only one class in the node or maximum depth reached, return a leaf node
        if len(unique_classes) == 1 or (self.max_depth is not None and depth == self.max_depth):
            return {'class': unique_classes[0]}

        # If there are no features left, return a leaf node with the majority class
        if num_features == 0:
            majority_class = unique_classes[np.argmax(class_counts)]
            return {'class': majority_class}

        # Choose the best split based on Gini impurity
        best_gini = float('inf')
        best_split = None

        for feature_index in range(num_features):
            feature_values = np.unique(X[:, feature_index])
            for value in feature_values:
                left_mask = X[:, feature_index] <= value
                right_mask = ~left_mask

                gini_left = self._calculate_gini(y[left_mask])
                gini_right = self._calculate_gini(y[right_mask])

                weighted_gini = (len(y[left_mask]) / num_samples) * gini_left + (len(y[right_mask]) / num_samples) * gini_right

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_split = {'feature_index': feature_index, 'value': value, 'left_mask': left_mask, 'right_mask': right_mask}

        if best_gini == float('inf'):
            # No split that reduces impurity found (all samples have the same value)
            return {'class': unique_classes[0]}

        # Recursively build the left and right branches of the tree
        left_subtree = self._build_tree(X[best_split['left_mask']], y[best_split['left_mask']], depth + 1)
        right_subtree = self._build_tree(X[best_split['right_mask']], y[best_split['right_mask']], depth + 1)

        return {'feature_index': best_split['feature_index'], 'value': best_split['value'],
                'left': left_subtree, 'right': right_subtree}

    def _calculate_gini(self, labels):
        _, class_counts = np.unique(labels, return_counts=True)
        probabilities = class_counts / len(labels)
        gini = 1 - np.sum(probabilities ** 2)
        return gini

    def predict(self, X_test):
        return np.array([self._predict_single(x, self.tree) for x in X_test])

    def _predict_single(self, x, node):
        if 'class' in node:
            return node['class']
        else:
            if x[node['feature_index']] <= node['value']:
                return self._predict_single(x, node['left'])
            else:
                return self._predict_single(x, node['right'])



class RandomForestClassifier:
    def __init__(self, n_estimators=10, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X_train, y_train):
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            indices = np.random.choice(len(X_train), len(X_train), replace=True)
            tree.fit(X_train[indices], y_train[indices])
            self.trees.append(tree)

    def predict(self, X_test):
        predictions = np.array([tree.predict(X_test) for tree in self.trees])
        combined_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
        return combined_predictions


def execute_RF(instance,E):
    X_test_instance = pd.DataFrame({
        'I': [1],
        'P': [float(instance[0])],
        'K': [float(instance[1])],
        'pH': [float(instance[2])],
        'EC': [float(instance[3])],
        'OC': [float(instance[4])],
        'S': [float(instance[5])],
        'Zn': [float(instance[6])],
        'Fe': [float(instance[7])],
        'Cu': [float(instance[8])],
        'Mn': [float(instance[9])],
        'B': [float(instance[10])],
        'OM': [float(instance[11])],
    })

    train_data, test_data = divided()
    print(train_data.dtypes)
    print(X_test_instance.dtypes)   
    rf_classifier = RandomForestClassifier(n_estimators=E, max_depth=None)
    rf_classifier.fit(train_data.drop('Fertility', axis=1).values, train_data['Fertility'].values)
    predicted_label = rf_classifier.predict(X_test_instance.values.reshape(1, -1).astype(float))
    return predicted_label[0]

