#!/usr/bin/env python
# coding: utf-8


import tensorflow as tf
import numpy as np
import datetime
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.utils import compute_class_weight

"""
    Ce modèle est un modèle de réseau de neurones convolutionnel (CNN) pour la classification. 

    Il utilise une approche de séquentiel en utilisant Keras.

    Le modèle commence par une couche de Conv1D, qui effectue la convolution sur une seule dimension. 
    
    La taille du noyau de convolution est de 3 et il y a 32 filtres. 
    
    La fonction d'activation utilisée est la ReLU (fonction d'activation rectifiée linéaire). 
    
    La forme de l'entrée est spécifiée à la première couche pour être de la forme (nombre de cas d'apprentissage,
    nombre de temps, nombre de signaux).

    La couche suivante est une couche de MaxPooling1D qui effectue une réduction de la dimension 
    en prenant le maximum de la sortie de la couche Conv1D sur une fenêtre définie. La taille de la fenêtre est 2.

    La couche suivante est une couche de Flatten, qui prend la sortie de la couche précédente 
    et la transforme en une seule dimension pour faciliter la connexion à une couche dense.

    Ensuite, il y a deux couches Dense, qui sont des couches complètement connectées. 
    
    La première couche a 128 neurones et la fonction d'activation est la ReLU. 
    
    La seconde couche a 6 neurones et la fonction d'activation est la Softmax, 
    qui renvoie une probabilité pour chaque classe.

    Le modèle est compilé en utilisant la fonction de perte "sparse_categorical_crossentropy" 
    pour les tâches de classification multi-classes et l'optimiseur "adam". 
    
    Les métriques suivies sont l'exactitude.

    Lors de l'entraînement, le modèle utilise les poids de classe équilibrés
    pour gérer les classes déséquilibrées en utilisant la fonction "compute_class_weight" de scikit-learn. 
    
    Le modèle est entraîné pendant 10 époques avec un lot de 100 échantillons et 
    les données de validation sont utilisées pour mesurer la performance.

"""

class EEGConvClassifier:
    def __init__(self):
        self.model = None
        self.class_weights = None
        
    def fit(self, X_train, y_train, X_val, y_val):
        self.model = Sequential()
        self.model.add(Conv1D(32, (3,), activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
        self.model.add(MaxPooling1D(2))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(6, activation='softmax'))

        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        checkpoint_cb = ModelCheckpoint('Al_Hajj',save_best_only=True)
        log_dir = "logs/fit_eeg_conv/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        classes = np.unique(y_train)
        self.class_weights = compute_class_weight(class_weight = "balanced", classes= classes, y= y_train)
        self.class_weights = {i: w for i, w in enumerate(self.class_weights)} 
        self.model.fit(X_train, y_train, epochs=10, batch_size=100, 
                       validation_data=(X_val, y_val), shuffle=True,class_weight=self.class_weights,
                       callbacks=[checkpoint_cb,tensorboard_callback])
        
    def predict(self, X_test):
        return self.model.predict_classes(X_test)

