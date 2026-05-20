from data_loader import load_normalized_data
from utils import train_test_split
from knn import KNN


def main():
    # 1. Load and normalize the data
    X, Y, scaler = load_normalized_data(file_path="bienetre.csv")

    # 2. Split into train / test sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_ratio=0.2, seed=42
    )
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    print("-" * 20)

    # 3. Train a baseline KNN with k=5
    print("Baseline KNN (k=5)")
    model = KNN(k=5)
    model.fit(X_train, Y_train)
    accuracy = model.evaluate(X_test, Y_test)
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Confusion matrix: {model.confusion_matrix(X_test, Y_test)}")
    print("-" * 20)

    # 4. Grid search over k
    print("Grid search over k in [1, 3, 5, 7, 9, 11, 15, 21]")
    result = KNN.grid_search(
        X_train, Y_train, X_test, Y_test,
        k_values=[1, 3, 5, 7, 9, 11, 15, 21],
    )
    print(f"\nBest k = {result['best_k']} "
          f"with accuracy {result['best_score']:.4f}")


if __name__ == "__main__":
    main()