from ANNExercise.NeuralNetwork.neural_network import *
from ANNExercise.NeuralNetwork.activation import *
from ANNExercise.NeuralNetwork.util import *
import scipy.io as sio

# Load data
data = sio.loadmat("MNIST.mat")
X = data["X"]
y = data["y"]
y = y.reshape(len(y))
y -= 1
X_train, y_train, X_test, y_test = train_valid_split(X, y, 0.7)
nn = NeuralNetwork()
nn.add(layer_name="Dense", activation="sigmoid", n_unit=400)
nn.add(layer_name="Dense", activation="ReLU", n_unit=512)
nn.add(layer_name="Dense", activation="ReLU", n_unit=128)
nn.add(layer_name="Dense", activation="softmax", n_unit=10)
nn.summary()
nn.fit(x_train=X_train, y_train=y_train, epochs=1500, learning_rate=0.7, batch_size=None, reg=0.003)
# plot_decision_boundary(nn.predict, X, y)
nn.accuracy_score(y_test, X_test)
nn.plot_history()
