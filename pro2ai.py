import numpy as np
from sklearn.metrics import confusion_matrix

import sys
import pandas as pd

from sklearn.metrics import pairwise_distances

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

TEST_SIZE = 0.3
K = 3


class NN:
    def __init__(self, trainingFeatures, trainingLabels) -> None:
        self.trainingFeatures = trainingFeatures
        self.trainingLabels = trainingLabels

    def fit(self, features, labels):
        self.trainingFeatures = np.array(features)
        self.trainingLabels = np.array(labels)

    def predict(self, features, k):
        test_features = np.array(features)

        distances = pairwise_distances(test_features, self.trainingFeatures,
                                       metric='euclidean')  # Compute pairwise distances between test and train features
        nearest_indices = np.argsort(distances)[:, :k]  # Get indices of k nearest neighbors for each test instance
        nearest_labels = np.take(self.trainingLabels, nearest_indices)  # Get labels of k nearest neighbors

        # Predict the majority class label among the k nearest neighbors for each test instance
        predicted_labels = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=nearest_labels)

        return predicted_labels.tolist()


def load_data(filename):
    df = pd.read_csv(filename)
    features = df.iloc[:, :-1].values.tolist()  # Extract all columns except the last one as features
    labels = df.iloc[:, -1].map({1: 1, 0: 0}).tolist()  # Map the last column to 1 if spam, 0 otherwise

    return features, labels


def preprocess(features):
    features = np.array(features)
    normalized_features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
    return normalized_features.tolist()


def train_mlp_model(features, labels):
    # Feature selection using SelectKBest with f_classif score function
    kbest_selector = SelectKBest(score_func=f_classif, k=57)
    features_selected = kbest_selector.fit_transform(features, labels)

    # Create and train an MLP classifier
    mlp = MLPClassifier(hidden_layer_sizes=(10, 5), activation="logistic", random_state=42)
    mlp.fit(features_selected, labels)

    return mlp


def evaluate(labels, predictions):
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    return accuracy, precision, recall, f1


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python safaa_1202065_duaa_1200909.py ./spambase.csv")

    # Load data from spreadsheet and split into train and test sets
    features, labels = load_data(sys.argv[1])
    features = preprocess(features)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=TEST_SIZE)

    # Train a k-NN model and make predictions
    model_nn = NN(X_train, y_train)
    predictions = model_nn.predict(X_test, K)
    accuracy, precision, recall, f1 = evaluate(y_test, predictions)

    # Print results
    print("**** k-NN Results ****")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    # Print the confusion matrix for k-NN results
    cm_knn = confusion_matrix(y_test, predictions)
    print("**** k-NN Confusion Matrix ****")
    print(cm_knn)
    # Train an MLP model and make predictions
    model = train_mlp_model(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy, precision, recall, f1 = evaluate(y_test, predictions)

    # Print results
    print("**** MLP Results ****")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    
    # Print the confusion matrix for MLP results
    cm_mlp = confusion_matrix(y_test, predictions)
    print("**** MLP Confusion Matrix ****")
    print(cm_mlp)

if __name__ == "__main__":
    main()
