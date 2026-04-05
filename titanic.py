# Titanic Survival Prediction using Machine Learning

import pandas as pd

df = pd.read_csv("titanic.csv")

# Drop unnecessary columns
df = df.drop(['Name','Ticket','Cabin','Embarked','PassengerId'], axis=1)

# Rename
df.rename(columns={'Sex': 'Gender'}, inplace=True)

# Convert categorical to numeric
df['Gender'] = df['Gender'].map({'male': 0, 'female': 1})

# Fill missing values
df['Age'] = df['Age'].fillna(df['Age'].median())

#splitting data
X = df.drop('Survived', axis=1)
y = df['Survived']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
#scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Create model
model = LogisticRegression()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))
