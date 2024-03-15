from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import joblib
from preprocess import load_dataset

def train_model():
    X, y = load_dataset('data/lfw')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pca = PCA(n_components=100, whiten=True, random_state=42)
    svm = SVC(kernel='linear', probability=True)

    model = make_pipeline(pca, svm)
    model.fit(X_train, y_train)

    print("Train accuracy:", model.score(X_train, y_train))
    print("Test accuracy:", model.score(X_test, y_test))

    joblib.dump(model, 'models/svm_model.pkl')

if __name__ == "__main__":
    train_model()
