import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

# Load the data
df = pd.read_csv("hippo-final-data.csv")

# Separate the features and the target
X = df.drop('encoded_story_types', axis=1)
y = df['encoded_story_types']

# Split into training, validation, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# Create a ColumnTransformer to handle text and numeric features
preprocessor = ColumnTransformer(
    transformers=[
        ('text1_tfidf', TfidfVectorizer(), 'story'),
        ('num', 'passthrough', ['distracted', 'draining', 'openness', 'Concreteness_Story', 'emotion_score_norm',
                               'num_sentences', 'avg_sentence_length', 'Sequentiality_FullSize', 'Topic_NLL_FullSize', 'fw_score_normalized'])
    ]
)

# Grid search parameters for Logistic Regression
param_grid = {
    'logisticregression__C': [0.01, 0.1, 1, 10, 100],
    'logisticregression__solver': ['lbfgs', 'saga']
}

# Values of k for cross-validation
cv_values = [3, 5, 7, 10]

# Iterate over different values of k
for k in cv_values:
    print(f"\nPerforming grid search with {k}-fold cross-validation.\n")

    # Create a pipeline with Logistic Regression
    pipeline = make_pipeline(
        preprocessor,
        LogisticRegression(random_state=42, max_iter=1000)
    )

    # Set up the grid search with k-fold cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=k, scoring='accuracy', verbose=1)

    # Train the model using grid search
    grid_search.fit(X_train, y_train)

    # Best parameters
    best_params = grid_search.best_params_
    print(f"Best parameters with {k}-fold CV: {best_params}")

# Get the best model from the grid search
best_model = grid_search.best_estimator_

# Predict and evaluate on the validation set
val_predictions = best_model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_predictions)
val_precision = precision_score(y_val, val_predictions, average='macro')
val_recall = recall_score(y_val, val_predictions, average='macro')
val_f1 = f1_score(y_val, val_predictions, average='macro')
val_conf_matrix = confusion_matrix(y_val, val_predictions)

print(f'\nValidation Metrics:')
print(f'Accuracy: {val_accuracy}')
print(f'Precision: {val_precision}')
print(f'Recall: {val_recall}')
print(f'F1 Score: {val_f1}')
print(f'Confusion Matrix:\n{val_conf_matrix}\n')

# Predict and evaluate on the test set
test_predictions = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
test_precision = precision_score(y_test, test_predictions, average='macro')
test_recall = recall_score(y_test, test_predictions, average='macro')
test_f1 = f1_score(y_test, test_predictions, average='macro')
test_conf_matrix = confusion_matrix(y_test, test_predictions)

print(f'\nTest Metrics:')
print(f'Accuracy: {test_accuracy}')
print(f'Precision: {test_precision}')
print(f'Recall: {test_recall}')
print(f'F1 Score: {test_f1}')
print(f'Confusion Matrix:\n{test_conf_matrix}')
