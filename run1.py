import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('symptomps.csv')

# Load data securely (call the load_data function)
symptom_data = pd.read_csv('symptomps.csv')
doctor_data = pd.read_csv('doctors.csv')

target_col = 'Disease'
feature_cols = symptom_data.drop(target_col,axis=1).columns
X_train, X_test, y_train, y_test = train_test_split(symptom_data[feature_cols], symptom_data[target_col], test_size=0.2)

le = LabelEncoder()
y_train = le.fit_transform(y_train)