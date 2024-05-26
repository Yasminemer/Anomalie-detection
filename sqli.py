import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Dense
from keras import regularizers
import tensorflow as tf
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope


# Constantes
LOG_FILE_PATH = "C:/Users/Yasmine/Desktop/PFE/logs2.txt"

# Étape 1 : Prétraitement des données
def extract_potential_sqli(logs):
    potential_sqli = []
    sql_injection_patterns = [
        r"(?i)\b(GET|SELECT|UPDATE|DELETE|INSERT|TRUNCATE|UNION)\b.*?\b(WHERE|SET|VALUES|AND|OR)\b.*?\b(OR|AND|NOT IN|LIKE|IN|=|<|>)\b",
        r"(?i)\b(GET|SELECT|UPDATE|DELETE|INSERT|TRUNCATE|UNION)\b.*?['\"].*?\b(OR|AND|NOT IN|LIKE|IN|=|<|>)\b\s+['\"].*?\b(OR|AND|NOT IN|LIKE|IN|=|<|>)\b",
        r"(?i)\b(OR|AND)\b\s*\d+\s*=\s*\d+",
        r"(?i)\b(OR|AND)\b\s+['\"].*?['\"]\s*=\s*['\"].*?['\"]",
        r"(?i)\b(OR|AND)\b\s+\w+\s*=\s*\w+",
        r"UNION\s+.+?SELECT",
        r"/.+?UNION.+?SELECT",
        r"'.+?OR\s+'.+?=",
        r"'.+?OR.+?=",
        r"--\s*[^-]+",
        r"/\*.*?\*/",
        r"load_file\("        
    ]
    for log in logs:
        for pattern in sql_injection_patterns:
            if re.search(pattern, log):
                potential_sqli.append(log)
                break
    return potential_sqli


# Étape 2 : Lecture du fichier de logs dans un DataFrame Pandas
logs_df = pd.read_csv(LOG_FILE_PATH, sep=";", header=None, names=['DateTime', 'Host', 'Type', 'Message'])

# Prétraitement des logs
potential_sqli = extract_potential_sqli(logs_df['Message'])

# Vectorisation des logs
vectorizer = TfidfVectorizer()
vectorizer.fit(potential_sqli)

# Étape 3 : Détection d'anomalies
def detect_anomalies(potential_sqli, vectorizer):
    X = vectorizer.transform(potential_sqli)

    # Isolation Forest
    isolation_forest = IsolationForest(n_estimators=500, contamination=0.00000000000000000000000000000000000000000000001)
    anomaly_scores_if = isolation_forest.fit_predict(X)
    anomalies_if_indices = np.where(anomaly_scores_if == 0)[0]
    anomalies_if = [potential_sqli[i] for i in anomalies_if_indices]

    # Plot des scores d'anomalie (Isolation Forest)
    plt.figure(figsize=(8, 6))
    plt.plot(anomaly_scores_if, 'bo', label='Isolation Forest')
    plt.title('Scores d\'anomalie (Isolation Forest)')
    plt.xlabel('Index du log')
    plt.ylabel('Score d\'anomalie')
    plt.legend()
    plt.show()

    # One-Class SVM
    one_class_svm = OneClassSVM(nu=0.1, kernel="rbf", gamma="auto")
    one_class_svm.fit(X)
    anomaly_scores_svm = one_class_svm.decision_function(X)
    anomalies_svm_indices = np.where(anomaly_scores_svm < 10 )[0]
    anomalies_svm = [potential_sqli[i] for i in anomalies_svm_indices]

    # Plot des scores d'anomalie (One-Class SVM)
    plt.figure(figsize=(8, 6))
    plt.plot(anomaly_scores_svm, 'ro', label='One-Class SVM')
    plt.title('Scores d\'anomalie (One-Class SVM)')
    plt.xlabel('Index du log')
    plt.ylabel('Score d\'anomalie')
    plt.legend()
    plt.show()

    # Elliptic Envelope
    elliptic_envelope = EllipticEnvelope(contamination=0.1)
    anomaly_scores_elliptic = elliptic_envelope.fit_predict(X.toarray())
    anomalies_elliptic_indices = np.where(anomaly_scores_elliptic == -1)[0]
    anomalies_elliptic = [potential_sqli[i] for i in anomalies_elliptic_indices]

    # Plot des scores d'anomalie (Elliptic Envelope)
    plt.figure(figsize=(8, 6))
    plt.plot(anomaly_scores_elliptic, 'yo', label='Elliptic Envelope')
    plt.title('Scores d\'anomalie (Elliptic Envelope)')
    plt.xlabel('Index du log')
    plt.ylabel('Score d\'anomalie')
    plt.legend()
    plt.show()
    # Local Outlier Factor (LOF)
    lof = LocalOutlierFactor(n_neighbors=50, contamination=0.5)
    anomaly_scores_lof = lof.fit_predict(X)
    anomalies_lof_indices = np.where(anomaly_scores_lof == -1)[0]
    anomalies_lof = [potential_sqli[i] for i in anomalies_lof_indices]

    # Plot des scores d'anomalie (LOF)
    plt.figure(figsize=(8, 6))
    plt.plot(anomaly_scores_lof, 'go', label='Local Outlier Factor')
    plt.title('Scores d\'anomalie (Local Outlier Factor)')
    plt.xlabel('Index du log')
    plt.ylabel('Score d\'anomalie')
    plt.legend()
    plt.show()

    return anomalies_lof,anomalies_if, anomalies_svm , anomalies_elliptic

# Étape 4 : Autoencoder

def train_autoencoder(potential_sqli, vectorizer):
    X = vectorizer.transform(potential_sqli)
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

    # Convertir les matrices de TF-IDF en SparseTensor
    X_train_sparse = tf.sparse.from_dense(X_train.toarray())
    X_val_sparse = tf.sparse.from_dense(X_val.toarray())

    # Réorganisation des données éparses pour garantir des indices correctement ordonnés
    X_train_reordered = tf.sparse.reorder(X_train_sparse)
    X_val_reordered = tf.sparse.reorder(X_val_sparse)

    # Convertir les SparseTensor en matrices denses
    X_train_reordered_dense = tf.sparse.to_dense(X_train_reordered)
    X_val_reordered_dense = tf.sparse.to_dense(X_val_reordered)

    input_dim = X_train_reordered_dense.shape[1]
    encoding_dim = 32

    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu', activity_regularizer=regularizers.l1(10e-5))(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = Model(input_layer, decoded)

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.fit(X_train_reordered_dense, X_train_reordered_dense, epochs=50, batch_size=32, shuffle=True, validation_data=(X_val_reordered_dense, X_val_reordered_dense))

    return autoencoder


def detect_anomalies_autoencoder(potential_sqli, vectorizer, autoencoder):
    X = vectorizer.transform(potential_sqli)

    # Convertir les matrices de TF-IDF en SparseTensor
    X_sparse = tf.sparse.from_dense(X.toarray())

    # Réorganisation des données éparses pour garantir des indices correctement ordonnés
    X_reordered_sparse = tf.sparse.reorder(X_sparse)

    # Prédiction avec l'autoencodeur
    anomaly_scores = autoencoder.predict(X_reordered_sparse)

    # Calcul des erreurs de reconstruction
    reconstruction_errors = np.sum(np.square(X.toarray() - anomaly_scores), axis=1)
    threshold = np.percentile(reconstruction_errors, 0.01)
    anomalies_ae_indices = np.where(reconstruction_errors > threshold)[0]
    anomalies_ae = [potential_sqli[i] for i in anomalies_ae_indices]

    # Plot des erreurs de reconstruction
    plt.figure(figsize=(8, 6))
    plt.plot(reconstruction_errors, 'bo', label='Erreurs de reconstruction')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Seuil')
    plt.title('Erreurs de reconstruction (Autoencoder)')
    plt.xlabel('Index du log')
    plt.ylabel('Erreur de reconstruction')
    plt.legend()
    plt.show()

    return anomalies_ae



# Détection d'anomalies avec Isolation Forest et One-Class SVM
anomalies_if, anomalies_svm, anomalies_lof, anomalies_elliptic= detect_anomalies(potential_sqli, vectorizer)

# Entraînement de l'Autoencoder
autoencoder = train_autoencoder(potential_sqli, vectorizer)

# Détection d'anomalies avec Autoencoder
anomalies_ae = detect_anomalies_autoencoder(potential_sqli, vectorizer, autoencoder)

print("Anomalies détectées (Isolation Forest):")
for anomaly in anomalies_if:
    print(anomaly)

print("\nAnomalies détectées (One-Class SVM):")
for anomaly in anomalies_svm:
    print(anomaly)

print("\nAnomalies détectées (Autoencoder):")
for anomaly in anomalies_ae:
    print(anomaly)
    
    


print("\nAnomalies détectées (Local Outlier Factor):")
for anomaly in anomalies_lof:
    print(anomaly)


print("\nAnomalies détectées (Elliptic Envelope):")
for anomaly in anomalies_elliptic:
    print(anomaly)
    
    
    

