from haar_filter import HaarFilter
import numpy as np


class WeakClassifier:

    def __init__(self, haar_filter):
        self.haar_filter = haar_filter
        self.theta = 0
        self.polarity = 1
        self.status = False

    def show(self):
        self.haar_filter.show()

    def train(self, X, Y, W):
        total_pos = np.sum(W[Y == 1])
        total_neg = np.sum(W) - total_pos
        scores = self.haar_filter.compute_on_image(X)
        scores = scores.flatten()
        Y = Y.flatten()
        data = sorted(list(zip(scores, Y, W)), key=lambda x: x[0])
        theta, polarity, best_error = None, None, np.inf
        seen_pos_weight, seen_neg_weight, error = 0, 0, 0
        for score, label, weight in data:
            error = min(seen_pos_weight + total_neg - seen_neg_weight, seen_neg_weight + total_pos - seen_pos_weight)
            if error < best_error:
                theta = score
                best_error = error
                polarity = -1 if seen_pos_weight >= seen_neg_weight else 1

            if label > 0:
                seen_pos_weight += weight
            else:
                seen_neg_weight += weight
        self.theta = theta
        self.polarity = polarity
        self.status = True
        return theta, polarity, best_error, scores[Y == 1], scores[Y == -1],

    def predict(self, X):
        if not self.status:
            print("Predicting using untrained weak-classifier.")
        scores = self.haar_filter.compute_on_image(X)
        prediction = np.ones(X.shape[0])
        return np.where(scores * self.polarity > self.polarity * self.theta, prediction, prediction - 2)