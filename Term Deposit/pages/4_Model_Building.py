import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score

@st.cache_data
def load_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    return train, test

train, test = load_data()

# Preprocess data
target = train['subscribed'].replace({'no': 0, 'yes': 1})
train = train.drop('subscribed', axis=1)
train = pd.get_dummies(train)
test = pd.get_dummies(test)

# Ensure all columns in test set are present in train set
missing_cols = set(train.columns) - set(test.columns)
for c in missing_cols:
    test[c] = 0
test = test[train.columns]  # align test set columns with train set

# Split data
X_train, X_val, y_train, y_val = train_test_split(train, target, test_size=0.2, random_state=12)

st.title('Model Building')

# Train and evaluate models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(max_depth=4, random_state=32),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=4, random_state=39),
    'K Nearest Neighbours': KNeighborsClassifier(n_neighbors=50),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, learning_rate=0.2, max_depth=4, random_state=53)
}

model_scores = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_val)
    accuracy = accuracy_score(y_val, predictions)
    precision = precision_score(y_val, predictions)
    f1 = f1_score(y_val, predictions)
    model_scores[model_name] = {'Accuracy': accuracy, 'Precision': precision, 'F1 Score': f1}

# Allow user to choose the model for final submission
chosen_model_name = st.selectbox('Choose the model for final submission', list(models.keys()))

if chosen_model_name:
    chosen_model = models[chosen_model_name]

    # Make predictions on the test dataset with the chosen model
    test_predictions = chosen_model.predict(test)

    # Calculate accuracy, precision, and F1 score for the chosen model on the validation set
    chosen_model_accuracy = model_scores[chosen_model_name]['Accuracy']
    chosen_model_precision = model_scores[chosen_model_name]['Precision']
    chosen_model_f1 = model_scores[chosen_model_name]['F1 Score']

    # Save predictions to a CSV file
    submission = pd.DataFrame({'ID': test.index, 'subscribed': test_predictions})
    submission['subscribed'].replace(0, 'no', inplace=True)
    submission['subscribed'].replace(1, 'yes', inplace=True)
    submission.to_csv('submission.csv', header=True, index=False)

    st.write(f'Predictions from {chosen_model_name} saved to submission.csv')
    st.write(f'### {chosen_model_name} Model Performance:')
    st.write(f'Accuracy: {chosen_model_accuracy:.4f}')
    st.write(f'Precision: {chosen_model_precision:.4f}')
    st.write(f'F1 Score: {chosen_model_f1:.4f}')
