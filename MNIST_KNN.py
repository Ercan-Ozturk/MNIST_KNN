


from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
mnist.keys()

X, y = mnist["data"], mnist["target"]
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

from sklearn.neighbors import KNeighborsClassifier
k_n = KNeighborsClassifier(weights='distance', n_neighbors=4)

k_n.fit(X_train, y_train)

predictions = k_n.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(predictions, y_test)
print(accuracy)





