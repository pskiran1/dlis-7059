import tensorflow as tf
import numpy as np
import os

# Define the input shape for numeric and string features
numeric_input_shape = (1,)
string_input_shape = (1,)

# Define the inputs
input_A = tf.keras.Input(shape=numeric_input_shape, name="A", dtype=tf.float32)
input_B = tf.keras.Input(shape=string_input_shape, name="B", dtype=tf.string)
input_C = tf.keras.Input(shape=numeric_input_shape, name="C", dtype=tf.float32)
input_D = tf.keras.Input(shape=numeric_input_shape, name="D", dtype=tf.float32)
input_E = tf.keras.Input(shape=numeric_input_shape, name="E", dtype=tf.float32)
input_F = tf.keras.Input(shape=string_input_shape, name="F", dtype=tf.string)
input_G = tf.keras.Input(shape=numeric_input_shape, name="G", dtype=tf.float32)
input_H = tf.keras.Input(shape=numeric_input_shape, name="H", dtype=tf.float32)
input_I = tf.keras.Input(shape=string_input_shape, name="I", dtype=tf.string)
input_J = tf.keras.Input(shape=numeric_input_shape, name="J", dtype=tf.float32)
input_K = tf.keras.Input(shape=numeric_input_shape, name="K", dtype=tf.float32)
input_L = tf.keras.Input(shape=string_input_shape, name="L", dtype=tf.string)
input_M = tf.keras.Input(shape=numeric_input_shape, name="M", dtype=tf.float32)
input_N = tf.keras.Input(shape=numeric_input_shape, name="N", dtype=tf.float32)
input_O = tf.keras.Input(shape=string_input_shape, name="O", dtype=tf.string)
input_P = tf.keras.Input(shape=numeric_input_shape, name="P", dtype=tf.float32)
input_Q = tf.keras.Input(shape=numeric_input_shape, name="Q", dtype=tf.float32)
input_R = tf.keras.Input(shape=string_input_shape, name="R", dtype=tf.string)
input_S = tf.keras.Input(shape=string_input_shape, name="S", dtype=tf.string)
input_T = tf.keras.Input(shape=numeric_input_shape, name="T", dtype=tf.float32)
input_U = tf.keras.Input(shape=numeric_input_shape, name="U", dtype=tf.float32)
input_V = tf.keras.Input(shape=numeric_input_shape, name="V", dtype=tf.float32)
input_W = tf.keras.Input(shape=numeric_input_shape, name="W", dtype=tf.float32)

# Concatenate all numeric inputs
all_numeric_inputs = tf.keras.layers.concatenate([
    input_A, input_C, input_D, input_E, input_G, input_H,
    input_J, input_K, input_M, input_N, input_P,
    input_Q, input_T, input_U, input_V, input_W
])

# Define a simple DNN
x = tf.keras.layers.Dense(128, activation='relu')(all_numeric_inputs)
x = tf.keras.layers.Dense(64, activation='relu')(x)
output = tf.keras.layers.Dense(1, activation='linear', name='output')(x)

# Create the model
model = tf.keras.Model(inputs=[
    input_A, input_B, input_C, input_D, input_E, input_F, input_G, input_H,
    input_I, input_J, input_K, input_L, input_M, input_N, input_O, input_P,
    input_Q, input_R, input_S, input_T, input_U, input_V, input_W
], outputs=output)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Save the model as a TensorFlow SavedModel
save_path = 'global_dnn/1/model.savedmodel'
model.save(save_path, save_format='tf')

print(f"Model saved to {save_path}")
