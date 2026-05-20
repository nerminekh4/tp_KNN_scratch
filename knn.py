class KNN:
    """
    k : int
        Number of neighbors used for the majority vote
    X_train : list[list[float]] | None
        Training feature matrix. None until `fit` is called
    Y_train : list | None
        Training labels. None until `fit` is called
    """
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.Y_train = None

    def __repr__(self):
        return f"KNN(k={self.k})"
    
    def __str__(self):
        if self.X_train is None:
            return f"KNN classifier (k={self.k}, not fitted yet)"
        return f"KNN classifier (k={self.k}, trained on {len(self.X_train)} samples)"
    
    @staticmethod
    def _euclidean_distance(point_a, point_b):
        """
        Compute the Euclidean distance between two equal-length vectors.

        d(a, b) = sqrt( sum_i (a_i - b_i)^2 )
        """
        squared_sum = 0.0
        for a, b in zip(point_a, point_b):
            squared_sum += (a - b) ** 2
        return squared_sum ** 0.5
    
    def fit(self, X_train, Y_train):
        self.X_train = [list(row) for row in X_train]
        self.Y_train = list(Y_train)
        if len(X_train) != len(Y_train):
            raise ValueError(
                f"X_train and Y_train must have the same length "
                f"(got {len(X_train)} and {len(Y_train)})"
            )
        if len(X_train) < self.k:
            raise ValueError(
                f"Cannot fit with fewer samples ({len(X_train)}) than k ({self.k})"
            )
        return self
    
    def _predict_one(self, x):
        distances = []
        for i, x_train in enumerate(self.X_train):
            d = self._euclidean_distance(x, x_train)
            distances.append((d, self.Y_train[i]))

        distances.sort(key=lambda pair: pair[0])
        k_nearest_labels = [label for _, label in distances[:self.k]]

        votes = {}
        for label in k_nearest_labels:
            votes[label] = votes.get(label, 0) + 1

        best_label = None
        best_count = -1
        for label, count in votes.items():
            if count > best_count:
                best_label = label
                best_count = count
        return best_label
    
    def predict(self, X_test):
        if self.X_train is None:
            raise RuntimeError("Model is not fitted yet. Call fit() first.")
        return [self._predict_one(list(x)) for x in X_test]
    
    def evaluate(self, X_test, Y_test):
        predictions = self.predict(X_test)
        Y_test_list = list(Y_test)
        if len(predictions) != len(Y_test_list):
            raise ValueError("predictions and Y_test must have the same length")
        correct = sum(1 for p, y in zip(predictions, Y_test_list) if p == y)
        return correct / len(Y_test_list)