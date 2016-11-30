import numpy as np
import json
from sklearn import svm
from sklearn.metrics import accuracy_score

def load(filename,name,numpy=False):
    """Carrega os dados de alguma rede anteriormente treinada."""
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    if numpy:
    	data_array = [np.array(w) for w in data[name]]
    	data_array = np.asarray(data_array)
    else:
    	data_array = [w for w in data[name]]
    return data_array


X = load("rootsift_input.json","input",True)
y = load("rootsift_output.json","output",True)

print X.shape
print y.shape

y = [y_.argmax() for y_ in y]

from sklearn import decomposition
pca = decomposition.PCA(n_components=10)
pca.fit(X)
X = pca.transform(X)

# randomizing positions

np.random.seed(42)
np.random.shuffle(X)

np.random.seed(42)
np.random.shuffle(y)

# spliting the dataset in thee groups
X_train = X[:6000]
y_train = y[:6000]

# X_validation = X[3000: 4000]
# y_validation = y[3000: 4000]

X_test = X[6000:]
y_test = y[6000:]

# creating the classifier
clf = svm.SVC()

# trainning the classifier with 9000 examples
clf.fit(X_train, y_train)


# verifying the accuracy for the model
predicted = clf.predict(X_test)
print accuracy_score(predicted, y_test)
predicted = clf.predict(X_train)
print accuracy_score(predicted, y_train)