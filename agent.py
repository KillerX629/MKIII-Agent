import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, Dense, Flatten, LSTM, Input, Reshape, Softmax, MultiHeadAttention
from keras.models import Model

# Define input shapes
input_shape = (4, 224, 256, 3)  # 4 frames of 224x256 RGB images
action_shape = (8,)  # 8 last controller inputs

# Define model inputs
input_frames = Input(shape=input_shape, name="input_frames")
input_actions = Input(shape=action_shape, name="input_actions")

# Define convolutional layers
conv1 = Conv2D(filters=32, kernel_size=3, strides=(
    2, 2), activation="relu")(input_frames)
conv2 = Conv2D(filters=64, kernel_size=3,
               strides=(2, 2), activation="relu")(conv1)
conv3 = Conv2D(filters=128, kernel_size=3,
               strides=(2, 2), activation="relu")(conv2)
flatten = Flatten()(conv3)
reshaped = Reshape((1, -1))(flatten)

# Define recurrent layers with attention mechanism
lstm = LSTM(256, return_sequences=True)(reshaped)
attn = MultiHeadAttention(num_heads=8, key_dim=256)(lstm, lstm)
attn_flatten = Flatten()(attn)

# Combine input and output layers
combined = keras.layers.concatenate([attn_flatten, input_actions])

# Define decision transformer
dense1 = Dense(256, activation="relu")(combined)
dense2 = Dense(128, activation="relu")(dense1)
# 12 possible actions in Mortal Kombat 3 Ultimate
output_actions = Dense(12, activation="softmax")(dense2)

# Define the model
model = Model(inputs=[input_frames, input_actions], outputs=output_actions)

# Compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.00025),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.save(filepath="models/AI-agent")