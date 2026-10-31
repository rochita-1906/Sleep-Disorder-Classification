
# Sleep Disorder Classification Using Machine Learning

An interactive machine learning application that predicts potential sleep disorder categories based on health and lifestyle data.

## Project Overview

Sleep disorders such as insomnia and sleep apnea can affect a person's health and quality of life. This project explores how machine learning can use lifestyle and health-related attributes to classify sleep disorder categories.

The project includes data preprocessing, model training, evaluation, and a web-based prediction interface built with Streamlit.

## Features

- Exploratory data analysis and data preprocessing
- Handling missing target values
- Encoding categorical features
- Training and comparing multiple machine learning algorithms
- Sleep disorder classification using a Voting Classifier
- Model evaluation using accuracy, precision, recall, and F1-score
- Interactive web application for user input and prediction
- Saved model and label encoders for reuse

## Machine Learning Models

The project explores the following algorithms:

- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Voting Classifier combining Random Forest and Decision Tree

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib

## Dataset

The project uses the Sleep Health and Lifestyle Dataset, which contains health and lifestyle attributes such as:

- Age and gender
- Occupation
- Sleep duration and sleep quality
- Physical activity level
- Stress level
- BMI category
- Blood pressure
- Heart rate
- Daily steps
- Sleep disorder category

Dataset source: [Sleep Health and Lifestyle Dataset](https://github.com/OludolapoAnalyst/Sleep-and-Lifestyle-Dataset)

## Project Structure

```text
SleepDisorderProject/
├── app.py
├── data.csv
├── sleep_disorder_model.pkl
├── label_encoders.pkl
├── README.md
├── .gitignore
└── venv/
```

The `venv/` directory is a local Python virtual environment and should not be uploaded to GitHub.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Sleep-Disorder-Classification.git
cd Sleep-Disorder-Classification
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit joblib
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

The application will open in your browser, usually at:

http://localhost:8501

## How to Use

1. Launch the Streamlit application.
2. Enter the requested health and lifestyle information.
3. Click **Predict Sleep Disorder**.
4. View the predicted sleep disorder category.
5. Review the model performance metrics displayed in the application.

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score

These metrics help assess the classification performance of the trained models.

## Disclaimer

This project is intended for educational and research purposes only. Its predictions are not medical diagnoses or a substitute for professional medical advice.

## Author

Developed as part of an internship project exploring machine learning applications in health and lifestyle data.
