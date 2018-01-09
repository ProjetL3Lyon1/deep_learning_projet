# Deep learning projet

## Installation et dependances

Installer Anaconda-navigator

* https://conda.io/docs/user-guide/install/index.html#installing-conda-on-a-system-that-has-other-python-installations-or-packages

Creer un environnement de travail sous Python 3.6

* ```conda create --name NAME python=3.6```

Installer les dépendances


* https://www.tensorflow.org/install/install_linux

Ex: pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.4.1-cp36-cp36m-linux_x86_64.whl

* ```pip install --upgrade keras```

* ```conda install scikit-image```

* ```conda install -c menpo opencv```

* ```sudo apt-get install python-tk python-imaging-tk python3-tk```

* ```pip install h5py```

* ```pip install Tensorboard```

## Présentation et utilisation

Pour voir les options disponibles:

* ```python main.py --help```

Pour lancer l'interface graphique Tkinter il suffit de taper sur la console.

* ```python main.py```

A travers l'interface graphique on peut :
* Générer une base de données d'images 
* Lancer l'entrainement sur la base d'images
* Tester une image pour prédire 

On veut que le réseau s’entraîne sur des images personnelles, nous mettons donc quelques
images de chaque dans le répertoire traitement-> im_base -> nom_de_dossier

Pour supprimer les images générées par la fonction de traitement,
il faut supprimer à la main les dossier im_learn et im_test dans le dossier traitement.

Pour lancer Tensorboard :

* ```tensorboard --logdir=tensorboard/```

Puis copier l'URL pour ensuite accéder à travers un navigateur l'interface graphique
de Tensorboard.
