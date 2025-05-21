import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# Dummy data
data = {
    'study_hours': [4, 2, 3, 1, 5],
    'attendance': [85, 60, 72, 55, 90],
    'internal_marks': [78, 50, 66, 40, 85],
    'sleep_hours': [6, 5, 7, 4, 8],
    'extracurricular': [1, 0, 0, 0, 1],
    'result': [1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)
X = df[['study_hours', 'attendance', 'internal_marks', 'sleep_hours', 'extracurricular']]
y = df['result']

model = LogisticRegression()
model.fit(X, y)

with open('student_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved as student_model.pkl")
