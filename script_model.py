import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def extract_features(file_path):
    df = pd.read_csv(file_path)
    
    df["timestamp_diff"] = df["record timestamp"].diff().fillna(0)
    df["distance"] = np.sqrt(df["x"].diff()**2 + df["y"].diff()**2).fillna(0)
    df["speed"] = df["distance"] / df["timestamp_diff"].replace(0, np.nan)
    df["accel"] = df["speed"].diff().fillna(0)
    
    avg_speed = df["speed"].mean()
    accel_var = df["accel"].var()
    jitter = df["distance"].std()
    smoothness = 1 / (1 + jitter)
    dwell_time = df[df["state"] == "Move"]["timestamp_diff"].mean()
    mid_crossings = ((df["x"] > df["x"].median()) & (df["x"].shift() < df["x"].median())).sum()

    return [avg_speed, accel_var, jitter, smoothness, dwell_time, mid_crossings]

train_folder = "training_files"
train_data = []
train_labels = []

for user_folder in os.listdir(train_folder):
    user_path = os.path.join(train_folder, user_folder)
    if os.path.isdir(user_path):
        for session_file in os.listdir(user_path):
            file_path = os.path.join(user_path, session_file)
            features = extract_features(file_path)
            train_data.append(features)
            train_labels.append(0)  

test_folder = "test_files"
test_data = []
test_sessions = []

for user_folder in os.listdir(test_folder):
    user_path = os.path.join(test_folder, user_folder)
    if os.path.isdir(user_path):
        for session_file in os.listdir(user_path):
            file_path = os.path.join(user_path, session_file)
            features = extract_features(file_path)
            test_data.append(features)
            test_sessions.append(session_file)

labels_df = pd.read_csv("public_labels.csv")
labels_dict = dict(zip(labels_df["filename"], labels_df["is_illegal"]))
test_labels = [labels_dict.get(session, 0) for session in test_sessions]

X_train = np.array(train_data)
y_train = np.array(train_labels)
X_test = np.array(test_data)
y_test = np.array(test_labels)

model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train, y_train)
joblib.dump(model, "captcha_model.pkl")

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
