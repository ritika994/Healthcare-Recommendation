import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Function to load data securely (replace with your implementation)
def load_data(data_path):
    try:
        data = pd.read_csv(data_path)
        return data
    except FileNotFoundError:
        print(f"Error: File '{data_path}' not found.")
        return None

def choose_model(model_type="classification"):
    if model_type == "classification":
        # Choose XGBoost for classification
        model = XGBClassifier(n_estimators=100, max_depth=5)
    elif model_type == "random_forest":
        # Choose RandomForestClassifier for classification
        model = RandomForestClassifier(n_estimators=100)
    else:
        raise ValueError("Invalid model type. Choose 'classification' or 'random_forest'.")

    return model

# Load data securely (call the load_data function)
symptom_data = load_data(r"C:\Users\ritik\Downloads\symptomps.csv.csv")
doctor_data = load_data(r"C:\Users\ritik\Downloads\doctors.csv.csv")

# Check if data loading was successful
if symptom_data is None or doctor_data is None:
    print("Error: Data loading failed. Please check file paths and try again.")
    exit()

# Define target variable
target = 'Disease'

# Define features
features = symptom_data.columns.drop(target).tolist()

# Separate numerical and categorical features
numerical_features = symptom_data[features].select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = symptom_data[features].select_dtypes(include=['object']).columns.tolist()

# Preprocessing pipeline for numerical features
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
    ('scaler', StandardScaler())  # Scale numeric features
])

# Preprocessing pipeline for categorical features
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values by filling with most frequent category
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
])

# Combine preprocessing steps for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Append classifier to preprocessing pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', choose_model())])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(symptom_data[features], symptom_data[target], test_size=0.2)

# Train the model
clf.fit(X_train, y_train)

# Evaluate the model
accuracy = clf.score(X_test, y_test)
print(f"Model accuracy: {accuracy}")

def analyze_symptoms(symptoms):
    # Preprocess user symptoms (similar to symptom data preprocessing)
    user_data = pd.DataFrame([symptoms], columns=features)

    # Make predictions based on the chosen model type
    predicted_disease = clf.predict(user_data)[0]

    # Filter doctors based on predicted disease
    recommended_doctors = doctor_data[doctor_data["Specialty"] == predicted_disease]

    # Find accuracy of the recommended doctors
    accuracy_dict = {}
    for index, doctor in recommended_doctors.iterrows():
        doctor_id = doctor['DoctorID']
        X_user = user_data.drop(columns=['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level'])
        X_user['DoctorID'] = doctor_id
        predicted_accuracy = clf.predict_proba(X_user)[0][1]
        accuracy_dict[doctor_id] = predicted_accuracy

    # Get the doctor with the highest accuracy
    max_accuracy_doctor_id = max(accuracy_dict, key=accuracy_dict.get)
    max_accuracy_doctor = recommended_doctors[recommended_doctors['DoctorID'] == max_accuracy_doctor_id]

    return max_accuracy_doctor['Name'].values[0], max_accuracy_doctor['Accuracy'].values[0]

# Example usage (replace with user input)
user_symptoms = ["fever", "cough", "fatigue"]

doctor_name, doctor_accuracy = analyze_symptoms(user_symptoms)
print("Recommended doctor:")
print(f"Name: {doctor_name}")
print(f"Accuracy: {doctor_accuracy}")
