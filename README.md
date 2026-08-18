# PremiumIQ — Health Insurance Premium Prediction

PremiumIQ is a Machine Learning-based application that predicts the **health insurance premium category** of an individual based on their health and demographic information.

The project uses Machine Learning techniques to analyze factors such as age, BMI, diabetes, blood pressure problems, chronic diseases, allergies, family history of cancer, and major surgeries to estimate the appropriate insurance premium category.

## 🚀 Project Overview

The main objective of PremiumIQ is to build an intelligent system that can help estimate health insurance premiums based on an individual's health profile.

The project includes:

* Data preprocessing and cleaning
* Feature engineering
* Exploratory Data Analysis (EDA)
* Machine Learning model training
* Hyperparameter tuning
* Model evaluation
* Model serialization
* Interactive prediction using Streamlit

## 🧠 Machine Learning Approach

The project was formulated as a **multi-class classification problem**, where the model predicts the appropriate `PremiumPrice` category.

### Features Used

The model uses health-related and demographic features such as:

* Age
* Height
* Weight
* BMI
* Diabetes
* Blood Pressure Problems
* Any Transplants
* Any Chronic Diseases
* Known Allergies
* History of Cancer in Family
* Number of Major Surgeries

Additional engineered features were created to improve the model, including:

* Health Risk Score
* Age × Health Risk
* Age × Surgeries
* Surgery × Transplant Risk

### Model

Several Machine Learning techniques were explored during the development of the project.

The final model uses:

**XGBoost Classifier**

Hyperparameter optimization was performed using techniques such as:

* RandomizedSearchCV
* GridSearchCV

The trained model was then saved using Python's model serialization libraries and integrated into the Streamlit application.

## 📊 Dataset

The project uses a health insurance dataset containing approximately **986 records**.

The dataset contains information related to an individual's demographic and medical profile along with their insurance premium category.

## 📈 Model Performance

The final model achieved approximately **93% R²/accuracy-level performance during the project evaluation**.

> Note: The exact evaluation metric should be mentioned here according to the final metric reported by the model. For a classification project, metrics such as Accuracy, Precision, Recall, F1-score, and Confusion Matrix are generally preferred.

## 🖥️ Application

The trained Machine Learning model is integrated into a **Streamlit web application**.

Users can enter their health information and receive a predicted insurance premium category.

### Application Workflow

```text
User Input
    ↓
Data Preprocessing
    ↓
Feature Engineering
    ↓
Trained XGBoost Model
    ↓
Premium Category Prediction
    ↓
Result Displayed in Streamlit
```

## 🛠️ Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* XGBoost
* TensorFlow / Keras
* CatBoost
* Ensemble Learning

### Data Processing & Visualization

* Pandas
* NumPy
* Matplotlib

### Deployment / Application

* Streamlit

### Model & Development Tools

* Joblib
* Pickle
* Git
* GitHub

## 📁 Project Structure

```text
PremiumIQ/
│
├── Medicalpremium.csv
├── model.pkl
├── app.py
├── requirements.txt
├── README.md
└── ...
```

> Update the structure above according to the actual files and folders in your repository.

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Nikhil-Darji02/PremiumIQ.git
```

### 2. Navigate to the project directory

```bash
cd PremiumIQ
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🔮 Future Improvements

Possible future improvements include:

* Using a larger and more diverse dataset
* Improving model interpretability
* Adding explainable AI techniques such as SHAP
* Improving the Streamlit UI/UX
* Deploying the application on a cloud platform
* Adding personalized insurance recommendations
* Comparing additional Machine Learning models

## 👨‍💻 Author

**Nikhil Darji**

Computer Engineering | Python | Machine Learning | AI

GitHub: [Nikhil-Darji02](https://github.com/Nikhil-Darji02)
