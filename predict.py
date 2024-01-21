import os
from keras.models import load_model
import numpy
import yaml

os.environ["KERAS_BACKEND"] = "tensorflow"

dataset = numpy.loadtxt("data/test_coffee_data.csv", delimiter=",", skiprows=1)

NIN = 5

X = dataset[:, 0:NIN]
Y = dataset[:, NIN:]

with open('scaler.yaml', 'r') as file:
    scaler = yaml.load(file, Loader=yaml.Loader)
    X_min_max_scaler = scaler['X_min_max_scaler']
    Y_min_max_scaler = scaler['Y_min_max_scaler']

X_scaled_0_1 = X_min_max_scaler.transform(X)
Y_scaled_0_1 = Y_min_max_scaler.transform(Y)

model = load_model("model_nn.keras")

predictions = model.predict(X_scaled_0_1)

inverse_prediction = Y_min_max_scaler.inverse_transform(predictions)

print("\nTest results:")
for i in range(len(Y)):
    print(f"{inverse_prediction[i][0]:.2f} (expected: {Y[i][0]})")

evaluate_history = model.evaluate(X_scaled_0_1, Y_scaled_0_1,  return_dict=True)
print("\nEvaluation metrics: ", evaluate_history)


