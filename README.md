# CECS 456 – Medical MNIST (6-class) CNN

This project implements a convolutional neural network (CNN) for classifying Medical MNIST images into 6 classes:

- AbdomenCT  
- BreastMRI  
- ChestCT  
- CXR (chest X-ray)  
- Hand  
- HeadCT  

The code trains a model on the Medical MNIST dataset and saves the trained model and evaluation results.

---

## 1. Project Structure

├─ medical_mnist_cnn.py        # Main training + evaluation script
├─ requirements.txt            # Python dependencies (TensorFlow, NumPy, etc.)
├─ results/                    # Saved model and result files (created by script)
│    accuracy.png
│    loss.png
│    confusion_matrix.png
│    confusion_matrix_normalized.png
│    classification_report.txt
│    medical_mnist_cnn.keras
└─ medical_mnist/              # Dataset folder (NOT included in the repo)
     AbdomenCT/
     BreastMRI/
     ChestCT/
     CXR/
     Hand/
     HeadCT/

2. Environment
This project was developed with:
- Python 3.12
- TensorFlow 2.x
- NumPy
- Matplotlib
- scikit-learn

You can install the main packages manually:
pip install tensorflow numpy matplotlib scikit-learn

3. Dataset Setup (Medical MNIST)
- Download the Medical MNIST dataset from Kaggle (dataset name: Medical MNIST – 58,954 medical images of 6 classes).
- Unzip the dataset.
- Inside this project folder, create a directory named medical_mnist and copy the class folders into it so the layout looks like:

medical_mnist/
    AbdomenCT/
    BreastMRI/
    ChestCT/
    CXR/
    Hand/
    HeadCT/

Each subfolder should contain its corresponding images.

4. How to Run
From the project root:
- (Optional) Create and activate a virtual environment.

- Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\activate.bat

-Install dependencies:
pip install tensorflow numpy matplotlib scikit-learn

Run the training script:
python medical_mnist_cnn.py

5. Outputs
When the script finishes, it will create/update the results/ directory with:
- medical_mnist_cnn.keras – saved Keras model.
- accuracy.png – training and validation accuracy curves.
- loss.png – training and validation loss curves.
- confusion_matrix.png – confusion matrix (counts).
- confusion_matrix_normalized.png – normalized confusion matrix.
- classification_report.txt – precision, recall, F1-score, and overall validation accuracy/loss.
