import joblib
from preprocess import load_dataset

def evaluate_model():
    model = joblib.load('models/svm_model.pkl')
    X_test, y_test = load_dataset('data/custom')  # sau alt test set
    acc = model.score(X_test, y_test)
    print(f"Evaluation Accuracy: {acc:.2%}")

if __name__ == "__main__":
    evaluate_model()
