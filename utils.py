import random


def train_test_split(X, Y, test_ratio=0.2, seed=42):
    """
    Split (X, Y) into training and test sets.

    Returns
    -------
    X_train, X_test, Y_train, Y_test : lists
    """
    X_list = [list(row) for row in X]
    Y_list = list(Y)

    n = len(X_list)
    if n != len(Y_list):
        raise ValueError("X and Y must have the same number of samples")

    indices = list(range(n))
    random.seed(seed)
    random.shuffle(indices)

    n_test = int(n * test_ratio)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    X_train = [X_list[i] for i in train_idx]
    X_test = [X_list[i] for i in test_idx]
    Y_train = [Y_list[i] for i in train_idx]
    Y_test = [Y_list[i] for i in test_idx]

    return X_train, X_test, Y_train, Y_test