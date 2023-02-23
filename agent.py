import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.optimizers import adam_v2
#from keras import layers

#imports para el agente
from rl.agents import DQNAgent
from rl.policy import BoltzmannGumbelQPolicy
from rl.memory import SequentialMemory


"""Tamaño de la entrada:
    224  320  3
    height, width, channels
    
número de acciones posibles:
    8
    [izq,der]
"""
#tf.image.rgb_to_grayscale(imagen) --> convierte una imagen a escala de grises

#np.reshape(imagen, (224, 320, 1)) --> convierte una imagen a una matriz de valores blanco y negro






def build_agent (model, actions_dim):
    policy = BoltzmannGumbelQPolicy()
    
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(policy=policy, memory=memory, model=model, nb_actions=actions_dim,
                   nb_steps_warmup=10, target_model_update=1e-2)
    return dqn






def build_model (actions_dim):
    
    #buscar como pasar de imagen de color a blanco y negro!
    inputImage = keras.Input(shape=(224, 320, 3))
    #las dimensiones de la entrada son entonces:
    # 224x320 y en vez de 3 canales (RGB), pasamos a blanco y negro
    
    inputPastInputs = keras.Input(shape=(10, 8))
    #esta entrada va a tener un vector con los ultimos 10 botones que apretó el agente, en orden cronológico
    #el 8 es por las acciones posibles
    
    #capas de convolución
    
    Conv = keras.layers.Conv2D(filters=32, kernel_size=(8, 8), strides=(4, 4), activation='relu') (inputImage)
    Conv = keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(2, 2), activation='relu') (Conv)
    Conv = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu') (Conv)
    Conv = keras.layers.Flatten()(Conv)
    Conv = keras.layers.Dense(units=512, activation='relu')(Conv)
    #en estas capas, el modelo debería interpretar la imagen del entorno actual del agente
    
    #interpretación de entradas pasadas:
    Flat = keras.layers.Flatten()(inputPastInputs)
    Combos = keras.layers.Dense(units=80)(Flat)
    Combos = keras.layers.Dense(units=32, activation='relu')(Combos)
    Combos = keras.layers.Dense(32, activation='relu')(Combos)
    #en estas capas, el modelo debería interpretar las acciones que el agente ha hecho en el entorno anterior
    #con un poco de suerte, el modelo aprenderá distintas combinaciones de botones para los poderes de un personaje
    
    #concatenación de las dos capas
    Out = keras.layers.concatenate([Conv, Combos])
    Out = keras.layers.Dense(units=64, activation='relu')(Out)
    Out = keras.layers.Dense(units=64, activation='relu')(Out)
    Out = keras.layers.Dense(units=actions_dim, activation='softmax')(Out)
    
    model = keras.Model(inputs=[inputImage, inputPastInputs], outputs=Out)
    #en esta capa, el modelo debería decidir qué acción tomar
    return model
    
    """"
if __name__ == "__main__":
    #keras.utils.plot_model(build_model(8),to_file='model.png', show_shapes=True)
    dqn = build_agent(build_model(8), 8)
    dqn.compile(adam_v2.Adam(learning_rate=1e-2), metrics=['mae'])
    """
    

#adam_v2 es un MODULO. El optimizador se accede mediante adam_v2.Adam

