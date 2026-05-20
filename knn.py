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