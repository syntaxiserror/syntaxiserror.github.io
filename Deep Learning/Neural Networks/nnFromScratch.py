import numpy as np
import nnfs 
from nnfs.datasets import spiral_data
nnfs.init()


class Layer_Dense:
	def __init__(self, n_inputs, n_neurons):
		self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
		self.biases = np.zeros((1, n_neurons))
	def forward(self, inputs):
		self.outcome = np.dot(inputs, self.weights) + self.biases


class ReLU:
	def forward(self, inputs):
		self.outcome = np.maximum(0, inputs)


class SoftMax:
	def forward(self, inputs):
		euler = np.exp(inputs - np.max(inputs, axis=1, keepdims=True)) 
		normals = euler / np.sum(euler, axis=1, keepdims=True)
		self.outcome = normals


class Loss:
	def calculate(self, output, y):
		sample_losses = self.forward(output, y)
		data_loss = np.mean(sample_losses)
		return(data_loss)


class Loss_CategoricalCrossEntropy(Loss):
	def forward(self, y_pred, y_true):
		samples = len(y_pred)
		y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

		if len(y_true.shape) == 1:
			correct_targets = y_pred_clipped[range(samples), y_true]
		elif len(y_true.shape) == 2:
			correct_targets = np.sum(y_pred_clipped*y_true, axis=1)
		
		negative_log = -(np.log(correct_targets))
		return negative_log
			

X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)
activation1 = ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = SoftMax()

dense1.forward(X)
activation1.forward(dense1.outcome)

dense2.forward(activation1.outcome)
activation2.forward(dense2.outcome)

print(activation2.outcome[:5])

loss_function = Loss_CategoricalCrossEntropy()
loss = loss_function.calculate(activation2.outcome, y)

print("Loss: ", loss)
