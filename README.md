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

## Presentation et utilisation

Plusieurs types d’utilisations, pour les consulter

* ```python main.py --help```

Lancer le programme sans argument permet de lancer l'interface graphique Tkinter.

On veut que le réseau s’entraîne sur des images personnelles, nous mettons donc quelques
images de chaque dans le répertoire traitement-> im_base -> nom_de_dossier

En lançant ```python main.py --database``` cela crée une base d'apprentissage et de test
pour que l'algorithme de Deep Learning puisse apprendre sur un tas d'image généré
par les images contenu dans im_base.

En lançant ```python main.py --training``` l'algorithme va s’entraîner sur la nouvelle
base.

En lançant ```python main.py --image image```  l'algorithme va analyser cette image
et tenter de trouver des correspondances avec ce qu'il a apprit en amont
