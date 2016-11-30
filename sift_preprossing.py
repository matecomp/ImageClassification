# import the necessary packages
from matplotlib import pyplot as plt
from rootsift import RootSIFT
import numpy as np
import os
import cv2

import json
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

def save(filename,name,archive):
    """Salva os valores do vies e dos pesos da rede num arquivo."""
    filename = filename
    data = {name: [v.tolist() for v in archive]}
    f = open(filename, "w")
    json.dump(data, f)
    f.close()

# 1) Load cifar_10 database
test_folder = "./img/cifar-10/test"
class_names = os.listdir(test_folder) # there are a folde for each class

# processing train folder
print "PROCESSING TEST FOLDER: "
X = []
cluster_data = []
y = []
count  = 0
# extract RootSIFT descriptors
rs = RootSIFT()
for name in class_names:
	files = os.listdir(test_folder+"/"+name)
	# transform each file into a feature vector using rootsift
	for file_name in files:
		image = cv2.imread(test_folder+"/"+name+"/"+file_name)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		detector = cv2.FeatureDetector_create("SIFT")
		kps = detector.detect(gray)

		# extract normal SIFT descriptors
		extractor = cv2.DescriptorExtractor_create("SIFT")
		(kps, descs) = extractor.compute(gray, kps)

		(kps, descs) = rs.compute(gray, kps)
		
		if descs is None:
			continue

		vec = descs
		X.append([v for v in vec])
		cluster_data.extend([v for v in vec])

		y_vec = [0] * len(class_names) # <<<<<<<<<<<<<< HOT ENCODING REPRESENTATION <<<<<
		y_vec[class_names.index(name)] = 1
		y.append(y_vec)

		count += 1

		if count % 1000 == 0:
			print count, " images processed"

print "Lost images: ", 10000-count

# after generate rootsift vector, we will compute the descriptor
# of an image by assigning each SIFT of the image to one of the 
# K clusters. In this way you obtain a histogram of length K.
from sklearn.cluster import KMeans
train = True
if train:
	n_clusters = 500
	kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(cluster_data)
	centers = kmeans.cluster_centers_
	X_pred = kmeans.labels_
else:
	centers = load("kmeans_centers.json","input",True)
	X_pred = load("kmeans_labels.json","input",True)
	n_clusters = centers.shape[0]

c = 0
newX = []
for x in X:
	aux = np.zeros((n_clusters))
	for desc in x:
		aux[X_pred[c]] += 1
		c += 1
	# mod = np.linalg.norm(aux)
	# aux = aux/aux.sum()
	newX.append(aux)

X = np.asarray(newX)
y = np.asarray(y)
print X.shape
print X[:10]

save("rootsift_input2.json","input",X)
save("rootsift_output2.json","output",y)
save("kmeans_centers2.json","centers",centers)
save("kmeans_labels2.json","labels",X_pred)