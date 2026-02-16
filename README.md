# Cancer_Prediction
This project builds a Neural Network model to classify breast tumors as Malignant (Cancerous) or Benign (Non-Cancerous) using the Breast Cancer Wisconsin (Diagnostic) dataset. The model is developed using TensorFlow Keras and achieves high prediction accuracy on unseen data.

# Dataset
Source: sklearn.datasets.load_breast_cancer()
Samples: 569
Features: 30 numerical medical attributes
Classes:
0 → Malignant
1 → Benign

# ⚙️Tech Stack

> Python
> Pandas
> NumPy
> Scikit-learn
> TensorFlow / Keras
> Matplotlib


## 🔍Project Workflow

#  1. Data Preparation
Loaded dataset from Scikit-learn
Converted to Pandas DataFrame
Added target label column

# 2. Exploratory Data Analysis
Checked dataset structure
Verified missing values
Analyzed class distribution

# 3. Data Preprocessing
Split data (80% training, 20% testing)
Applied feature scaling using StandardScaler

# 4. Model Development
Built Sequential Neural Network
Hidden Layer: 20 neurons (ReLU activation)
Output Layer: 2 neurons (Sigmoid activation)
Optimizer: Adam
Loss Function: sparse_categorical_crossentropy
Trained for 10 epochs

# 5. Model Evaluation
Achieved ~95% test accuracy
Visualized training and validation performance

# 6. Prediction
Implemented prediction system for new tumor samples  


# 📈 Results
Test Accuracy: ~95%
Model generalizes well on unseen data

# Future Improvements

Add confusion matrix and classification report,
Hyperparameter tuning,
Compare with traditional ML models,
Deploy as a web application, 

# 👨‍💻 Author
Shikhar Mishra
Engineering Student | Data Analytics Enthusiast | AI Learner
