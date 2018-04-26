import os, random
from PIL import Image
import numpy as np
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import DistanceMetric
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier

x = []
y = []
count1 = 0
count0 = 0

#-------------------------------Generate Model and Hyperparameters----------------------------
def TPOT_Classifier():  
	tpot = TPOTClassifier(verbosity=2, max_time_mins=390, population_size=40,)
	tpot.fit(x_train, y_train)
	tpot.export('tpot_assignment_pipeline.py')
	TPOT_predict = tpot.predict(x_test)
	score = tpot.score(x_test, y_test)
	print(score)
	print(y_test)
	print(TPOT_predict)
	return score

#----------------------Accuracy score from Model used as Metric Function----------------------
def Metric(x, y):
	return score

#---------------------------------------KNN Classifier---------------------------------------
def scikit_learn():
	DistanceMetric.get_metric('pyfunc', func=Metric)
	KNN = KNeighborsClassifier(n_neighbors=2, algorithm='auto', metric=Metric)
	KNN.fit(x_train, y_train)
	predic = KNN.predict(x_test)
	print(accuracy_score(y_test, predic))
	print(y_test)
	print(predic)

#-------------------------Model and hyperparameters from TPOT pipeline-----------------------
def TPOT_model():
	model = RandomForestClassifier(bootstrap=False, class_weight="balanced",criterion="gini", 		max_features=0.05, min_samples_leaf=7, min_samples_split=8,n_estimators=100)
	model.fit(x_train, y_train)
	predic = model.predict(x_test)
	score = accuracy_score(y_test, predic)
	print(score)
	print(y_test)
	print(predic)
	return score

#--------------------------------Load A Sample of Random data---------------------------------
for i in range (0, 10000):
	language = random.choice(os.listdir("/home/mlvm2/ce888lab/Assignment/omniglot-master/python/images_background"))
	#print(language)

	filename1 = random.choice(os.listdir("/home/mlvm2/ce888lab/Assignment/omniglot-master/python/images_background/"+language))
	#print(filename1)
	image1 = random.choice(os.listdir("/home/mlvm2/ce888lab/Assignment/omniglot-master/python/images_background/"+language+"/"+filename1))
	#print(image1)

	filename2 = random.choice(os.listdir("/home/mlvm2/ce888lab/Assignment/omniglot-master/python/images_background/"+language))
	#print(filename2)
	image2 = random.choice(os.listdir("/home/mlvm2/ce888lab/Assignment/omniglot-master/python/images_background/"+language+"/"+filename2))
	#print(image2)

	if filename1 == filename2:
		dif = 0
	else:
		dif = 1
	#print(dif)

	im1 = Image.open("/home/mlvm2/ce888lab/Assignment/omniglot-master/python/images_background/"+language+"/"+filename1+"/"+image1)
	im2 = Image.open("/home/mlvm2/ce888lab/Assignment/omniglot-master/python/images_background/"+language+"/"+filename2+"/"+image2)

	#im1.show()
	image1array = list(im1.getdata())
	image2array = list(im2.getdata())

	if count1<500 and dif == 1:
		xadd = (np.array([image1array,image2array])).ravel()
		yadd = np.array([dif])
		x.append(xadd)
		y.append(yadd)
		count1 = count1 + 1
	if count0<500 and dif == 0:
		xadd = (np.array([image1array,image2array])).ravel()
		yadd = np.array([dif])
		x.append(xadd)
		y.append(yadd)
		count0 = count0 + 1

x = np.asarray(x)
y = np.asarray(y)

#---------------------------------Converts 2D array to 1D array-------------------------------
y = y.ravel()

#------------------------------------------Split Data-----------------------------------------

x_train, x_test, y_train, y_test = train_test_split(x, y,train_size=0.75, test_size=0.25) 
#print("Train x: \n",x_train,"\nTrain y: \n",y_train,"\nTest x: \n",x_test,"\nTest y: \n",y_test)
#print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

#------------------------------------------Functions------------------------------------------
#score = TPOT_Classifier()
score = TPOT_model()
scikit_learn()


