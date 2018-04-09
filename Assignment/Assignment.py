#! /usr/bin/python3
import os, random
from PIL import Image
import numpy as np

x = []
y = []

for i in range (0, 20):
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

	xadd = np.array([image1array,image2array])
	yadd = np.array([dif])
	x.append(xadd)
	y.append(yadd)

x = np.asarray(x)
y = np.asarray(y)
print(x)
print(y)

#create two numpy arrays one with images in (x) and one with same or different (y)
# [image 1, image 2,			[ 1,
#  image 1, image 2]			  0 ]

