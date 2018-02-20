#! /usr/bin/python3
import os, random

language = random.choice(os.listdir("/home/mlvm2/ce888lab/Assignment/omniglot-master/python/images_background"))
print(language)

filename1 = random.choice(os.listdir("/home/mlvm2/ce888lab/Assignment/omniglot-master/python/images_background/"+language))
print(filename1)
image1 = random.choice(os.listdir("/home/mlvm2/ce888lab/Assignment/omniglot-master/python/images_background/"+language+"/"+filename1))
print(image1)

filename2 = random.choice(os.listdir("/home/mlvm2/ce888lab/Assignment/omniglot-master/python/images_background/"+language))
print(filename2)
image2 = random.choice(os.listdir("/home/mlvm2/ce888lab/Assignment/omniglot-master/python/images_background/"+language+"/"+filename2))
print(image2)

if filename1 == filename2:
	dif = 0
else:
	dif = 1
print(dif)

