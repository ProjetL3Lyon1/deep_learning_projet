import os
import skimage
import numpy as np
from skimage import io, exposure, img_as_uint, img_as_float
from skimage.transform import rotate, resize

#_________________ Creation des dossiers pour les images d'apprentissage et les images de test___________________

mkdirLearn = "mkdir -p ./traitement/im_learn/"
mkdirTest = "mkdir -p ./traitement/im_test/"
os.system(mkdirLearn)
os.system(mkdirTest)

#_________________Copie des repertoires de im_base dans im_learn et im_test_______________________________________

listDir = os.listdir('./traitement/im_base') # on liste les repertoires presents dans im_base

for directory in listDir:

	mkdirLearnDir = "mkdir -p ./traitement/im_learn/"+directory
	mkdirTestDir = "mkdir -p ./traitement/im_test/"+directory
	os.system(mkdirLearnDir)
	os.system(mkdirTestDir)

#_______________________________________Traitement des images_____________________________________________________

	listFiles = os.listdir('./traitement/im_base/'+directory) # on liste les fichiers presents dans les repertoires
	k = 0
	print (directory)
	for files in listFiles: # on traite les images 1 par 1

		filename = os.path.join('./traitement/im_base/'+directory,files)


		v_min = 0.5 # on initialise les valeurs de l'intensite lumineuse a [0.5 , 0.8], doit etre compris entre [0.0 , 1.0]
		v_max = 0.8

		for l in range (2):

			# on va generer les images avec 5 niveaux de luminosite differente
			for j in range (5):

				# mais egalement avec 10 niveaux de rotation differente
				for i in range(5):

					image = io.imread(filename)
					image = img_as_float(image)

					if (i==1):
						image = rotate(image, (-5), resize=0) 
					if (i==2):
						image = rotate(image, 5, resize=0)
					if (i==3):
						image = rotate(image, (-10), resize=0)
					if (i==4):
						image = rotate(image, 10, resize=0)

					image = resize(image, (200, 200), mode='reflect') # on resize l'image en 200x200
					image = exposure.rescale_intensity(image, in_range=(v_min,v_max)) # et on applique la modification d'intensite lumineuse

					if (l==1):
						image = np.fliplr(image)

					num = int(str(k)+str(l)+str(i)+str(j))
					if (num%4==0):
						io.imsave('./traitement/im_test/'+directory+'/image'+str(num)+'.png', img_as_uint (image)) # on garde 1/4 des images pour la phase de test
					else:
						io.imsave('./traitement/im_learn/'+directory+'/image'+str(num)+'.png', img_as_uint (image)) # on sauvegarde l'image, nommee avec les indices de chaque boucle

				v_min-=0.1 # on decremente la v_min de 0.1
			k+=1

print("SAVE")
