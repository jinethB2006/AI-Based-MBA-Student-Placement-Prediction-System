# AI-Based MBA Student Placement Prediction System

## Overview

The AI-Based MBA Student Placement Prediction System is an end-to-end Machine Learning web application that predicts whether an MBA student is likely to be placed based on academic performance, educational background, employability test score, work experience, and specialization.

The project is developed using Python, Scikit-learn, Flask, HTML, and CSS. Four Machine Learning algorithms were trained and evaluated, and Logistic Regression was selected as the final model for deployment.

---

## Features

- Predicts MBA student placement status
- Interactive web interface using Flask
- Data preprocessing and feature engineering
- One-Hot Encoding for categorical features
- Feature Scaling using StandardScaler
- Comparison of multiple Machine Learning models
- Logistic Regression used as the final prediction model
- Clean and responsive user interface

---

## Technologies Used

- Python
- Flask
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib
- HTML
- CSS

---

## Machine Learning Models Used

- Logistic Regression ✅ (Final Model)
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

---

## Dataset Features

### Numerical Features

- SSC Percentage
- HSC Percentage
- Degree Percentage
- Employability Test Percentage
- MBA Percentage

### Categorical Features

- Gender
- SSC Board
- HSC Board
- HSC Stream
- Degree Type
- Work Experience
- Specialisation

### Target Variable

- Placement Status (Placed / Not Placed)

---

## Project Workflow

1. Data Collection
2. Data Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Encoding
5. Feature Scaling
6. Model Training
7. Model Evaluation
8. Model Selection
9. Model Saving using Joblib
10. Flask Web Application Development
11. Prediction Deployment

---

## Project Structure

```
AI-Based-MBA-Student-Placement-Prediction-System/
│
├── dataset/
│
├── model/
│   ├── student_placement_logistic_model.pkl
│   ├── scaler.pkl
│   ├── encoder.pkl
│   └── feature_names.pkl
│
├── notebook/
│   └── placement_prediction.ipynb
│
├── static/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Model Performance

| Model | Accuracy |
|-------|----------|
| Logistic Regression | **86.05%** |
| Decision Tree | 69.77% |
| Random Forest | 86.05% |
| Support Vector Machine (SVM) | 81.40% |

**Final Selected Model:** Logistic Regression

---

## Installation

### Install Required Packages

```bash
pip install -r requirements.txt
```

### Run the Flask Application

```bash
python app.py
```

Open your browser and visit:

```
http://127.0.0.1:5000
```

---

## Future Improvements

- Add prediction probability score
- Improve the user interface
- Deploy the application on Render or Railway
- Integrate a database for prediction history
- Support additional placement datasets
- Include more placement-related features for broader applicability

---

## Note

This project is developed using an MBA student placement dataset. Therefore, the prediction model is intended for MBA students who can provide all the required academic and placement-related information.