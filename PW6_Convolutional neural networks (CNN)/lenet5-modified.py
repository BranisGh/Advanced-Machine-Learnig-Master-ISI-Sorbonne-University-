# Réseau inspiré de http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# Pour les utilisateurs de MacOS  (pour utiliser plt & keras en même temps)
import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'


def displayConvFilers(model, layer_name, num_filter=4, filter_size=(3,3), num_channel=0, fig_size=(2,2)):
    
    layer_dict = dict([(layer.name, layer) for layer in mnist_model.layers])
    
    weight, biais = layer_dict[layer_name].get_weights()
    print(weight.shape) 
    plt.figure(figsize=fig_size)
    for i in range(num_filter):
        plt.subplot(fig_size[0],fig_size[1],i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        vis = np.reshape(weight[:,:,num_channel,i],filter_size)
        plt.imshow(vis, cmap=plt.cm.binary)
    plt.show()

## Chargement et normalisation des données
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

# POUR LES CNN : On rajoute une dimension pour spécifier qu'il s'agit d'imgages en NdG
train_images = train_images.reshape(60000,28,28,1)
test_images = test_images.reshape(10000,28,28,1)


# One hot encoding
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

filter_size_conv1 = (5,5)

## Définition de l'architecture du modèle
mnist_model = tf.keras.models.Sequential()
# Expliquez à quoi correspondent les valeurs numériques qui définissent les couches du réseau
mnist_model.add(tf.keras.layers.Conv2D(filters=6,kernel_size=filter_size_conv1,padding="same", activation='tanh', input_shape=(28, 28, 1)))
mnist_model.add(tf.keras.layers.AveragePooling2D())
mnist_model.add(tf.keras.layers.Conv2D(filters=16,kernel_size=(5,5),padding="valid", activation='tanh'))
mnist_model.add(tf.keras.layers.AveragePooling2D())
mnist_model.add(tf.keras.layers.Flatten())
mnist_model.add(tf.keras.layers.Dense(120 , activation='tanh'))
#mnist_model.add(tf.keras.layers.Dropout(0.5))
mnist_model.add(tf.keras.layers.Dense(84 , activation='tanh'))
mnist_model.add(tf.keras.layers.Dense(10 , activation='softmax'))

# expliquer le nombre de paramètre de ce réseau
print(mnist_model.summary())


sgd = tf.keras.optimizers.Adam()

mnist_model.compile(sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# On visualise avant l'entrainement
displayConvFilers(mnist_model, 'conv2d', 
                    num_filter=6, 
                    filter_size=filter_size_conv1, 
                    num_channel=0, 
                    fig_size=(2,3)
                )



mnist_model.fit(train_images,
         train_labels,
         batch_size=64,
         epochs=20
         )

## Evaluation du modèle 
test_loss, test_acc = mnist_model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

displayConvFilers(mnist_model,
                    'conv2d', 
                    num_filter=6, 
                    filter_size=filter_size_conv1, 
                    num_channel=0, 
                    fig_size=(2,3)
                )



'''
displayConvFilers(mnist_model, 'conv2d_1', 
                    num_filter=16, 
                    filter_size=(5,5), 
                    num_channel=1, 
                    fig_size=(4,4)
                )
'''