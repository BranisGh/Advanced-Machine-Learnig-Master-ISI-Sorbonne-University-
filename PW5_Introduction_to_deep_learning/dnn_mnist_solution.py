## RESULTAT des différentes architectures sur 10 epochs (seed non fixée)
# MLP 25 neurones + sigmoid + mean_squared_error + SGD : 0.4459
# MLP 25 neurones + sigmoid + categorical_crossentropy + SGD : 0.8854
# MLP 25 neurones + sigmoid + softmax (sortie) + categorical_crossentropy + SGD : 0.8957
# MLP 25 neurones + relu + softmax (sortie) + categorical_crossentropy + SGD : 0.9251
# MLP 200 neurones + relu + softmax (sortie) + categorical_crossentropy + SGD : 0.9412
# MLP 100 / 50 / 25 neurones + relu + softmax (sortie) + categorical_crossentropy + SGD: 0.9558
# MLP 100 / 50 / 25 neurones + relu + softmax (sortie) + categorical_crossentropy + SGD + BN: 0.9686
# MLP 200 neurones + relu + softmax (sortie) + categorical_crossentropy + Adam : 0.9792
# MLP 100 / 50 / 25 neurones + relu + softmax (sortie) + categorical_crossentropy + Adam: 0.9769
# MLP 100 / 50 / 25 neurones + relu + softmax (sortie) + categorical_crossentropy + Adam + BN: 0.9776

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# Pour les utilisateurs de MacOS  (pour utiliser plt & keras en même temps)
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

## Chargement des données
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Observez la dimension des données. A quoi correspondent ces dimensions ?
train_images.shape

# Récupérez la dimension des images
height = train_images.shape[1]
width = train_images.shape[2]

# Affichez la première image de la base d'apprentissage
im = train_images[0,:,:]
#plt.imshow(im)
#plt.show() 
# Affichez son label associé
print('label : {}'.format(train_labels[0]))


## Préparation des données
# Normalisation des données
# pourquoi divise-t-on par 255.0 ?
train_images = train_images / 255.0
#train_images = train_images.astype('float32') / 255  # équivalent
test_images = test_images / 255.0

#Peut-on utiliser directement les images en entrée du MLP ? Pourquoi ?
#train_images = np.reshape(train_images,(train_images.shape[0],height*width))
#test_images = np.reshape(test_images,(test_images.shape[0],height*width))

# One hot encoding
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

# Observer les nouvelles dimension des données d'apprentissage
train_labels.shape

## Définition du modèle

## Définition de l'architecture du modèle
mnist_model = tf.keras.models.Sequential()
# Expliquez à quoi correspondent les valeurs numériques qui définissent les couches du réseau
mnist_model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
mnist_model.add(tf.keras.layers.Dense(100 ,input_shape=(784,),  activation='relu'))
mnist_model.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

mnist_model.add(tf.keras.layers.Dense(50 ,input_shape=(100,),  activation='relu'))
mnist_model.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

mnist_model.add(tf.keras.layers.Dense(25 ,input_shape=(50,),  activation='relu'))
mnist_model.add(tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

mnist_model.add(tf.keras.layers.Dense(10 , activation='softmax'))


sgd = tf.keras.optimizers.Adam()  #SGD(lr=0.001, momentum=0.9) 

mnist_model.compile(sgd, loss='categorical_crossentropy', metrics=['accuracy'])

mnist_model.fit(train_images,
         train_labels,
         batch_size=64,
         epochs=10)


## Evaluation du modèle 
test_loss, test_acc = mnist_model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)


# visualiser la couche de poids
# Observer les différences en fonction de la méthode d'optimisation et de l'architecture du réseau
# layers[1] pour passer le flatten et arriver sur la première couche
# get_weights()[0] -> les poids
# get_weights()[1] -> les biais
mnist_model.layers[1].get_weights()[1].shape
weights_all  = mnist_model.layers[1].get_weights()[0]

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    vis = np.reshape(weights_all[:,i],(height,width))
    plt.imshow(vis, cmap=plt.cm.binary)
    #plt.xlabel(class_names[train_labels[i]])
plt.show()

# visualisation du nombre de paramètres du réseau
mnist_model.summary()



