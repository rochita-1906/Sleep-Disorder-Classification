import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier

# Page setup
st.set_page_config(
    page_title="Sleep Disorder Prediction",
    page_icon="🌙",
    layout="centered"
)

st.title("🌙 Sleep Disorder Prediction")
st.write("Enter your sleep and lifestyle details below.")

# Load dataset
df = pd.read_csv("data.csv")
df = df.drop("Person ID", axis=1)
df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")

# Encode categorical data
categorical_columns = [
    "Gender", "Occupation", "BMI Category",
    "Blood Pressure", "Sleep Disorder"
]

encoders = {}

for column in categorical_columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column].astype(str))
    encoders[column] = encoder

# Prepare training data
X = df.drop("Sleep Disorder", axis=1)
y = df["Sleep Disorder"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train voting model
model = VotingClassifier(
    estimators=[
        ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ("dt", DecisionTreeClassifier(random_state=42))
    ],
    voting="hard"
)

model.fit(X_train, y_train)
import joblib

# Save the trained model and encoders
joblib.dump(model, "sleep_disorder_model.pkl")
joblib.dump(encoders, "label_encoders.pkl")

print("Model saved successfully!")
from sklearn.metrics import accuracy_score, classification_report

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Performance")
st.metric("Test Accuracy", f"{accuracy * 100:.2f}%")

st.write("Classification Report:")
report = classification_report(
    y_test,
    y_pred,
    target_names=encoders["Sleep Disorder"].classes_,
    output_dict=True,
    zero_division=0
)

st.dataframe(pd.DataFrame(report).transpose())

# User input form
with st.form("prediction_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 80, 30)

    occupations = sorted(df["Occupation"].unique())
    occupation_names = encoders["Occupation"].inverse_transform(occupations)
    occupation = st.selectbox("Occupation", occupation_names)

    sleep_duration = st.slider("Sleep Duration (hours)", 3.0, 12.0, 7.0)
    quality = st.slider("Quality of Sleep (1-10)", 1, 10, 7)
    activity = st.slider("Physical Activity Level", 0, 100, 50)
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)

    bmi = st.selectbox(
        "BMI Category",
        encoders["BMI Category"].classes_
    )

    blood_pressure = st.selectbox(
        "Blood Pressure",
        encoders["Blood Pressure"].classes_
    )

    heart_rate = st.slider("Heart Rate", 40, 150, 70)
    steps = st.number_input("Daily Steps", 0, 30000, 7000)

    submitted = st.form_submit_button("Predict Sleep Disorder")

if submitted:
    input_data = pd.DataFrame([{
        "Gender": encoders["Gender"].transform([gender])[0],
        "Age": age,
        "Occupation": encoders["Occupation"].transform([occupation])[0],
        "Sleep Duration": sleep_duration,
        "Quality of Sleep": quality,
        "Physical Activity Level": activity,
        "Stress Level": stress,
        "BMI Category": encoders["BMI Category"].transform([bmi])[0],
        "Blood Pressure": encoders["Blood Pressure"].transform(
            [blood_pressure]
        )[0],
        "Heart Rate": heart_rate,
        "Daily Steps": steps
    }])

    prediction = model.predict(input_data)[0]
    disorder = encoders["Sleep Disorder"].inverse_transform([prediction])[0]

    st.subheader("Prediction Result")
    st.success(f"Predicted category: {disorder}")

    st.info(
        "This is an educational machine learning prediction, "
        "not a medical diagnosis. Consult a healthcare professional "
        "if you have sleep concerns."
    )