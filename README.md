# Face Recognition System (PCA + SVM)

🗓️ **March 2024**

This project implements a face recognition pipeline using:
- PCA for feature reduction
- SVM for classification
- Grayscale + Histogram Equalization + Resize for preprocessing

## Dataset
- [LFW (Labeled Faces in the Wild)](http://vis-www.cs.umass.edu/lfw/)
- Custom dataset support (place images in `data/custom/`)

## Usage

```bash
pip install -r requirements.txt
python main.py
```

## Model Output
- Trained model saved as `models/svm_model.pkl`
