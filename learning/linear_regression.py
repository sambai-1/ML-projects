import numpy as np
from numpy.typing import NDArray


def get_model_prediction(X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.round(np.matmul(X, weights), 5)

def get_error(model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
    tmp = np.mean(np.square(model_prediction - ground_truth))
    return np.round(tmp, 5)

class Training:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self, 
        X: NDArray[np.float64], 
        Y: NDArray[np.float64], 
        num_iterations: int, 
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        for i in range(num_iterations):
            prediction = self.get_model_prediction(X, initial_weights)
            d1 = self.get_derivative(prediction, Y, len(X), X, 0)
            d2 = self.get_derivative(prediction, Y, len(X), X, 1)
            d3 = self.get_derivative(prediction, Y, len(X), X, 2)
            initial_weights[0] = initial_weights[0] - d1 * self.learning_rate
            initial_weights[1] = initial_weights[1] - d2 * self.learning_rate
            initial_weights[2] = initial_weights[2] - d3 * self.learning_rate
        # you will need to call get_derivative() for each weight
        # and update each one separately based on the learning rate!
        # return np.round(your_answer, 5)
        return np.round(initial_weights, 5)
        pass