import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, IsolationForest

# Charger les données de journaux dans un DataFrame Pandas
logs = pd.read_csv("C:/Users/Yasmine/Desktop/PFE/logs.txt", sep=';', names=['timestamp', 'host', 'service', 'log'], parse_dates=['timestamp'], date_parser=lambda x: datetime.strptime(x, '[%Y/%m/%d %H:%M:%S]'))

# Prétraiter les données
logs['is_bruteforce'] = logs['log'].str.contains('Failed password for Yasmine@RTI.local from|Failed brute force attempt for Yasmine@RTI.local', case=False).astype(int)

# Diviser les données en X et y
X = logs[['host', 'service']]
le = LabelEncoder()
X['host'] = le.fit_transform(X['host'])
X['service'] = le.fit_transform(X['service'])
y = logs['is_bruteforce']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner les différents modèles
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
mlp.fit(X_train, y_train)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

gb = GradientBoostingClassifier()
gb.fit(X_train, y_train)

xgb = XGBClassifier(random_state=42)
xgb.fit(X_train, y_train)

# Entraîner le modèle Isolation Forest
isoforest = IsolationForest(random_state=42)
isoforest.fit(X_train)

# Évaluer les performances sur les données de test pour chaque modèle
models = [rf, mlp, dt, gb, xgb, isoforest]
model_names = ['Random Forest', 'ANN', 'Arbre de décision', 'Gradient Boosting', 'XGBoost', 'Isolation Forest']
accuracies = []

for model, name in zip(models, model_names):
    if name == 'Isolation Forest':
        y_pred = (isoforest.predict(X_test) == -1).astype(int)
    else:
        y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    logs[f'{name}_prediction'] = model.predict(X)  # Ajouter une colonne pour chaque modèle

    print(f"\n{name} Résultats :")
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
    # Visualisation de la matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    plt.imshow(cm, cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.title(f"Matrice de confusion - {name}")
    plt.xlabel("Prédiction")
    plt.ylabel("Réalité")
    plt.show()



# Trouver le meilleur modèle
best_model_idx = accuracies.index(max(accuracies))
best_model_name = model_names[best_model_idx]
print(f"\nLe meilleur modèle est : {best_model_name}")