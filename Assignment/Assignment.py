import os, random
from PIL import Image
import numpy as np
from tpot import TPOTClassifier
import csv
x = []
y = []

def split(array):
	split_num = int(len(array)/(10/7))
	return array[:split_num], array[split_num:]

for i in range (0, 30):
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
#print(x)
#print(y)

y = y.ravel()

x_train, x_test = split(x)
y_train, y_test = split(y)

#print("Train x: \n",x_train,"\nTrain y: \n",y_train,"\nTest x: \n",x_test,"\nTest y: \n",y_test)
#print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

tpot = TPOTClassifier(verbosity=2, max_time_mins=5, population_size=40)
tpot.fit(x_train, y_train)
print(tpot.score(x_test, y_test))

tpot.export('tpot_asignment_pipeline.py')

