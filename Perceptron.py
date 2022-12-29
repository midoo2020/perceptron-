import numpy as np

def predict(X, weights):
    # Calculer la somme pondérée des entrées
    z = np.dot(X, weights)
    # Retourner +1 si la somme est supérieure à zéro, -1 sinon
    return np.where(z > 0, 1, -1)

def fit(X, y, learning_rate, n_iter):
    # Initialiser les poids à des valeurs aléatoires
    weights = np.random.rand(X.shape[1])
    
    for _ in range(n_iter):
        # Faire des prédictions
        predictions = predict(X, weights)
        # Calculer l'erreur
        error = y - predictions
        # Mettre à jour les poids
        weights += learning_rate * np.dot(error, X)
    
    return weights

# Préparer les données d'entraînement et de test
X_train = ...
y_train = ...
X_test = ...
y_test = ...

# Entraîner le modèle en utilisant les données d'entraînement
weights = fit(X_train, y_train, learning_rate=0.1, n_iter=100)

# Évaluer la performance du modèle sur les données de test
predictions = predict(X_test, weights)
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.2f}")

# Faire des prédictions sur de nouvelles données
X_new = ...
predictions = predict(X_new, weights)
