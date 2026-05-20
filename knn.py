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