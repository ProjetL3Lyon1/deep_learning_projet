# coding: utf-8
#----------------------------------------------------------------------------------------
#Importation des modules et lib
import skimage
import os
from skimage.transform import rotate, resize
import numpy as np
from skimage import io, exposure, img_as_uint, img_as_float

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from time import time
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.preprocessing import image
import argparse
import sys

from tkinter import * 
from functools import partial
from tkinter import filedialog
from tkinter import ttk
from tkinter.ttk import *

#----------------------------------------------------------------------------------------

#Gestion des arguments en mode console
#exemple pour lancer l'entrainement: python main.py --training 
parser = argparse.ArgumentParser()
parser.add_argument('--database', action='store_true', help='Genere une base de donnee')
parser.add_argument('--training', action='store_true', help='S\'entraine sur la base de donnee')
parser.add_argument('--image', nargs='?', help='Verifie la reconaissance de cette image')
parser.add_argument('--tkinter', action='store_true', help='Interface graphique')
args = parser.parse_args()

#dimensions des images.
img_width, img_height = 200, 200


#le nombre d'itération 
epochs = 5
#le nombre d'échantillon dans chaque itération
batch_size = 16
folder = ''
h5 = ''


# ----------------------------------------------------------------------------------------

if not sys.argv[1:]:
    parser.print_help()

# ----------------------------------------------------------------------------------------

def database():
	"""Fonction qui genère la base de donnée"""
	print("Creation de la base de donnee")
	os.system("python -W ignore ./traitement/traitement_image.py")
	print("Fin de la creation de la base de donnee")
	affichage(modeleCree)

if (args.database):
	database()
    

# ----------------------------------------------------------------------------------------

def training(valepoch, valbatch, nomModele):
	"""
	Fonction qui permet à l'ia de s'entrainer 
	sur la base de données d'images.
	"""
	epochs=valepoch.get()
	batch_size=valbatch.get()
	h5=nomModele.get()

	nb_dossier_learn = int(os.popen('ls ./traitement/im_learn | wc -l').read())
	nb_dossier_base = int(os.popen('ls ./traitement/im_base | wc -l').read())
	if (nb_dossier_learn < nb_dossier_base):
		print ("Referez vous a l'aide, -h")
	else:
		print("Debut de l'exercice")

		train_data_dir = 'traitement/im_learn'
		validation_data_dir = 'traitement/im_test'

		nb_train_samples = int(os.popen('find ./traitement/im_learn -type f | wc -l').read())
		nb_validation_samples = int(os.popen('find ./traitement/im_test -type f | wc -l').read())
		num_classes = len(sorted(os.listdir('./traitement/im_learn')))

		if K.image_data_format() == 'channels_first':
			input_shape = (3, img_width, img_height)
		else:
			input_shape = (img_width, img_height, 3)

		#----------------------------------------------------------------

		''''
		Configuration du réseau de neurones.
		Le modèle utilise ici les couches convolutifs bidimensionnelle.
		'''
		model = Sequential()

		#configuration 1
		model.add(Conv2D(32, (3, 3), input_shape=input_shape))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))

		#configuration 2
		model.add(Conv2D(32, (3, 3)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.5))

		#configuration 3
		model.add(Conv2D(64, (3, 3)))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.5))

		#configuration 4
		model.add(Flatten())
		model.add(Dense(64))
		model.add(Activation('relu'))
		model.add(Dropout(0.5))
		'''
		Construction de la sortie avec num_classes neurones, 
		un pour chaque classe cible.
		'''
		model.add(Dense(num_classes))
		model.add(Activation('softmax'))
		model.compile(optimizer='rmsprop',
					loss='categorical_crossentropy',
					metrics=['accuracy'])
		#----------------------------------------------------------------
		#utilisation de tensorboard
		#tensorboard --logdir=tensorboard/
		tensorboard = TensorBoard(log_dir="tensorboard/{}".format(time()),
									write_graph=False, write_images=True)

		'''
		Génèration de lots de données d'image avec 
		une augmentation de données en temps réel.
		Les données seront bouclées (par lots) indéfiniment.
		'''
		train_datagen = ImageDataGenerator(
								rescale=1. / 255,
								shear_range=0.2,
								zoom_range=0.2,
								horizontal_flip=True)

		test_datagen = ImageDataGenerator(rescale=1. / 255)

		train_generator = train_datagen.flow_from_directory(
						train_data_dir,
						target_size=(img_width, img_height),
						batch_size=batch_size,
						class_mode='categorical')

		validation_generator = test_datagen.flow_from_directory(
						validation_data_dir,
						target_size=(img_width, img_height),
						batch_size=batch_size,
						class_mode='categorical')

		model.fit_generator(
					train_generator,
					steps_per_epoch=nb_train_samples // batch_size,
					epochs=epochs,
					callbacks=[tensorboard],
					validation_data=validation_generator,
					validation_steps=nb_validation_samples // batch_size)

		# score = model.evaluate_generator(validation_generator, nb_validation_samples)
		# print('Test Loss:', score[0])
		# print('Test accuracy:', score[1])
		print ("Fin de l'exercice")

		# Sauvegarde du modèle
		model.save(h5+'.h5')
		print ("Enregistrement du fichier "+h5+".h5")
		modeleCree = True 
		affichage(modeleCree)
# ----------------------------------------------------------------------------------------

if (args.training):
	valepoch=0
	valebatch = 0
	nomModele = ""
	training(valepoch, valebatch, nomModele)

    

# ----------------------------------------------------------------------------------------

def fctImage():
	"""Fonction de prédiction sur une image donnée"""
	print("Analyse de l'image")
	model = load_model(nomModele.get()+'.h5')

	img = image.load_img(folder, target_size=(img_width, img_height))
	x = image.img_to_array(img)
	x/= 255
	x = np.expand_dims(x, axis=0)

	classes = model.predict_classes(x)
	prediction = model.predict(x)[0]
	listDir = sorted(os.listdir('./traitement/im_learn'))
	# print(listDir)
	# print(prob)
	i=0
	for y in listDir:
		print(y+' : '+str(100*prediction[i])+' %')
		i+=1
	# print ()
	i = 0
	for directory in listDir:
		if i == classes:
			print ("Forte chance que ce soit : "+directory)
			text10.set('\nResultat : '+directory)
			break
		else:
			i+=1

	# print()
	print("Fin de l'analyse de l'image")


if (args.image):
    image()

    # --------------
#def restart_program():
#	python = sys.executable
#	os.exec1(python, python, * sys.argv)

#_______________________________________________AFFICHAGE________________________________________________

def affichage(modeleCree):
	"""Fonction qui affiche l'interface graphique du programme"""
	
	boolvar1 = BooleanVar()
	boolvar1.set(False)
	listDir = os.listdir('./traitement')
	compteur = 0

	for directory in listDir:
		compteur = compteur +1
		
	if (compteur<3):
		text1.set("\nAucune base d'entrainement n'a été générée\nà partir de votre base personnelle d'images.\n")
		l1.pack()
		b2.config(state=DISABLED)
		text7.set("\nVous ne pouvez pas lancer l'apprentissage\nsans base d'entrainement")
		l7.pack()

	else:
		b1.config(state=DISABLED)
		
		text1.set("\nUne base d'entrainement à été trouvée.\n")
		l1.pack()
		
		text2.set("Elle a été formée à partir de votre base personnelle\ncontenant les classes suivantes :\n")
		l2.pack()
		
		listDir2 = os.listdir('./traitement/im_base')
		text3.set(str(listDir2))
		l3.pack()

		b2.config(state=NORMAL)
		text7.set(" ")
		l7.pack()


	if (modeleCree == True):
		b4.config(state=NORMAL)
		text9.set("\nModèle :"+str(nomModele.get())+"\n")
	else :
		b4.config(state=DISABLED)
		text9.set("\nVous n'avez pas encore\ncréé de modèle.\n")
		


#_______________________________________________PARCOURIR________________________________________________

def parcourir():
	"""Fonction qui permet de parcourir un fichier"""
	global folder
	folder = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("jpeg files","*.jpg"),("png files","*.png")))
	filename.set(folder)
	



#_______________________________________________FENETRE________________________________________________


fenetre = Tk()
fenetre.style = Style()
fenetre.style.theme_use("clam")#('clam', 'alt', 'default', 'classic')
fenetre.title("Projet Deep Learning")
fenetre['bg']='bisque'

modeleCree = False

#_______________________________________________FRAME 1________________________________________________

frame1 = LabelFrame(fenetre, text="Génération d'une base ", borderwidth=30,relief=GROOVE) #bg="white",
frame1.pack(fill="both", expand="yes", side=LEFT,padx=30, pady=30)

b1 = Button(frame1, text="Générer la base d'entrainement", command = lambda: database())
#b1.grid(column=0, row=1)
b1.pack()

text1 = StringVar()
l1 = Label(frame1, textvariable=text1)

text2 = StringVar()
l2 = Label(frame1, textvariable=text2)

text3 = StringVar()
l3 = Label(frame1, textvariable=text3)


#_______________________________________________FRAME 2________________________________________________

frame2 = LabelFrame(fenetre, text="Training : phase d'apprentissage", borderwidth=30, relief=GROOVE)#, padx=20, pady=20)
frame2.pack(fill="both", expand="yes", side=LEFT, padx=30, pady=30)

l4 = Label(frame2, text="Choisir le nombre d'epochs : ")
l4.pack()
valepoch = IntVar()
s = Spinbox(frame2, textvariable=valepoch, from_=1, to=100)
s.pack()

l5 = Label(frame2, text="\nChoisir le batch size : ")
l5.pack()
valbatch = IntVar()
s1 = Spinbox(frame2, textvariable=valbatch, from_=8, to=64, increment=8)
s1.pack()

l6 = Label(frame2, text="\nSauvegarder le modèle sous le nom : ")
l6.pack()
nomModele = StringVar() 
nomModele.set("")
entree = Entry(frame2, textvariable=nomModele, width=30)
entree.pack()

b2 = Button(frame2, text="Lancer l'apprentissage", command = partial(training, valepoch, valbatch, nomModele))
#b2.grid(column=0, row=1)
b2.pack()

text7 = StringVar()
l7 = Label(frame2, textvariable=text7)


#_______________________________________________FRAME 3________________________________________________

frame3 = LabelFrame(fenetre, text="Test : image/modèle", borderwidth=30, relief=GROOVE)#, padx=20, pady=20)
frame3.pack(fill="both", expand="yes", side=RIGHT, padx=30, pady=30)

l8 = Label(frame3, text="Chargez une image pour tester\nle modèle que vous avez créé :\n")
l8.pack()

b3 = Button(frame3,text='Parcourir',command=lambda: parcourir())
b3.pack()

filename = StringVar(frame3)
entry = Entry(frame3, textvariable=filename)
entry.pack()

text9 = StringVar()
l9 = Label(frame3, textvariable=text9)
l9.pack()

b4 = Button(frame3, text="Lancer le test", command = lambda: fctImage())
b4.pack()


text10 = StringVar()
text10.set('\nResultat : ')
l10 = Label(frame3, textvariable=text10)
l10.pack()

affichage(modeleCree)

fenetre.mainloop()
