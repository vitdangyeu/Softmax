import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from autils import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Softmax function
def my_softmax(z):
    e_z = np.exp(z)
    softmax = e_z/(np.sum(e_z))
    return softmax

# Load dataset
X, y = load_data()

# Visualizing the Data

m, n = X.shape

fig, axes = plt.subplots(8, 8, figsize=(8, 10))
fig.tight_layout(pad=0.13, rect = [0, 0.03, 1, 0.91])

for i, ax in enumerate(axes.flat):
    # Select random indices
    random_index = np.random.randint(m)

    # Reshape the image
    X_random_reshaped = X[random_index].reshape((20, 20)).T

    # Display the image
    ax.imshow(X_random_reshaped, cmap='gray')

    # Display the label above the image
    ax.set_title(f"{y[random_index, 0]}")
    ax.set_axis_off()
    fig.suptitle("Label, image", fontsize = 16)
plt.show()

# Model representation
tf.random.set_seed(1234)
model = Sequential([
    tf.keras.Input(shape = 400),
    Dense(units = 25, activation='relu'),
    Dense(units = 15, activation='relu'),
    Dense(units = 10, activation='linear'),
])
model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
)
history = model.fit(
    X, y,
    epochs=40
)

# Loss
plot_loss_tf(history)

# Prediction
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

m, n = X.shape

fig, axes = plt.subplots(8, 8, figsize=(8, 10))
fig.tight_layout(pad=0.13, rect = [0, 0.03, 1, 0.91])

for i, ax in enumerate(axes.flat):
    # Select random indices
    random_index = np.random.randint(m)

    # Reshape the image
    X_random_reshaped = X[random_index].reshape((20, 20)).T

    # Display the image
    ax.imshow(X_random_reshaped, cmap='gray')

    # Prediction
    prediction = model.predict(X[random_index].reshape(1,400))
    prediction_p = tf.nn.softmax(prediction)
    y_hat = np.argmax(prediction_p)

    # Display the label above the image
    ax.set_title(f"{y_hat, y[random_index, 0]}")
    ax.set_axis_off()
    fig.suptitle("Label, image", fontsize = 16)
plt.show()
