import os, random
from PIL import Image
import numpy as np
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

x = []
y = []

def TPOT_Classifier():  
	#Split Data
	x_train, x_test, y_train, y_test = train_test_split(x, y,train_size=0.75, test_size=0.25) 

	#print("Train x: \n",x_train,"\nTrain y: \n",y_train,"\nTest x: \n",x_test,"\nTest y: 	   	\n",y_test)
	print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

	#TPOT Classifier
	tpot = TPOTClassifier(verbosity=2, max_time_mins=5, population_size=40)
	tpot.fit(x_train, y_train)
	print(tpot.score(x_test, y_test))
	tpot.export('tpot_assignment_pipeline.py')

def scikit_learn():
	KN = KNeighborsClassifier(n_neighbors=2)
	KN.fit(x, y)

#--------------------------------Load A Sample of Random data---------------------------------
for i in range (0, 50):
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

	xadd = (np.array([image1array,image2array])).ravel()
	yadd = np.array([dif])
	x.append(xadd)
	y.append(yadd)

x = np.asarray(x)
y = np.asarray(y)

#Converts 2D array to 1D array
y = y.ravel()
#-------------------------------------------------------------------------------------------

TPOT_Classifier()
scikit_learn()

