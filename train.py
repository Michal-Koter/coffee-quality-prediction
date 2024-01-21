import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import yaml
import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KERAS_BACKEND"] = "tensorflow"


MODEL_FILENAME = "model_nn.keras"
VAL_SPLIT = 0.2
EPOCHS = 100
HIDDEN_UNITS = 50
BATCH_SIZE = 128
NIN = 5


def build(hidden_units=HIDDEN_UNITS):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(NIN,)))
    model.add(keras.layers.Dense(hidden_units, activation='relu'))
    model.add(keras.layers.Dense(10, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.build()
    return model


if __name__ == "__main__":
    dataset = np.loadtxt("data/train_coffee_data.csv", delimiter=",", skiprows=1)

    X = dataset[:, 0:NIN]
    Y = dataset[:, NIN:]

    X_min_max_scaler = MinMaxScaler().fit(X)
    Y_min_max_scaler = MinMaxScaler().fit(Y)
    X_scaled_0_1 = X_min_max_scaler.transform(X)
    Y_scaled_0_1 = Y_min_max_scaler.transform(Y)

    data = {'X_min_max_scaler': X_min_max_scaler,
            'Y_min_max_scaler': Y_min_max_scaler}
    with open('scaler.yaml', "w") as file:
        yaml.dump(data, file)

    model = build()
    model.compile(loss='mean_squared_error', optimizer='adam') #, metrics=['r2_score'])
    H = model.fit(X_scaled_0_1, Y_scaled_0_1, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2)

    model.save(MODEL_FILENAME)

    plt.figure()
    plt.plot(range(EPOCHS), H.history["loss"], label="Train loss")
    plt.plot(range(EPOCHS), H.history["val_loss"], label="Validation loss")
    # plt.plot(range(EPOCHS), H.history["r2_score"], label="Train r2_score")
    # plt.plot(range(EPOCHS), H.history["val_r2_score"], label="Validation r2_score")
    plt.title("Train / validation loss")
    plt.xlabel("Epoch #")
    plt.ylabel("R2")
    plt.legend(loc="lower left")
    plt.savefig("model.png")


