'''
The only way to make it work is to use PCA 🤔
'''

import pandas as pd
import pylab as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.datasets import make_regression
from sklearn.decomposition import PCA

# load data
spectra = pd.read_csv('Absorption_spectra_set.csv')
concentration = pd.read_csv('Concentration_set.csv')

# Generate some example data
# Replace this with your actual data
x_data = spectra.drop('wavelength', axis=1).T
y_data = concentration['concentration_lactate'] * 100

# x_data, y_data = make_regression(n_samples=400, n_features=350, noise=0.8, random_state=42)
# x_data = pd.DataFrame(x_data)
# y_data = pd.DataFrame(y_data)

# Data normalization
scaler = StandardScaler()
x_data = scaler.fit_transform(x_data)
# x_data = spectra.drop('wavelength', axis=1).T.pct_change(axis=1).drop(0, axis=1)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)

# pca = PCA(40)
# pca.fit(x_train)
# x_train = pca.transform(x_train)
# x_test = pca.transform(x_test)


# define r-squared (lets use it as in other models)
def r_squared(y_true, y_pred):
    SS_res = tf.keras.backend.sum(tf.keras.backend.square(y_true - y_pred))
    SS_tot = tf.keras.backend.sum(tf.keras.backend.square(y_true - tf.keras.backend.mean(y_true)))
    return 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())


# Create an RNN model
def init_model():

    input = tf.keras.Input(shape=(x_train.shape[1], 1))
    x = tf.keras.layers.SimpleRNN(units=16, kernel_regularizer=tf.keras.regularizers.l2(0.01))(input)
    y_ = tf.keras.layers.Dense(1, activation='relu', use_bias=False)(x)

    model = tf.keras.Model(inputs=input, outputs=y_)
    model.compile(
        loss=['mean_squared_error'],
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0025),
        metrics=[r_squared],
    )

    return model


# Train the model
model = init_model()
model.summary()
history = model.fit(x_train, y_train, epochs=2000, batch_size=x_train.shape[0], validation_data=(x_test, y_test))

# plot results
y_train.reset_index(drop=True).plot()
pd.Series(model.predict(x_train).ravel()).plot()

# Evaluate the model
print(model.layers[2].weights, model.layers[1].weights)
y_ = model.predict(x_test)
loss, r_squared_score = model.evaluate(x_test, y_test)
print("Mean Squared Error on Test Set:", loss)
print("R-squared on Test Set:", r_squared_score)

# Plot training history
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['r_squared'], label='Training R-squared')
plt.plot(history.history['val_r_squared'], label='Validation R-squared')
plt.title('R-squared vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('R-squared')
plt.ylim(0, 1)
plt.legend()

plt.show()