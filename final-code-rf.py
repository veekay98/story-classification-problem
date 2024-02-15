import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

# Load data from a CSV file into a DataFrame
df = pd.read_csv("hippo-final-data.csv")

# Separate the features (X) and target variable (y)
X = df.drop('encoded_story_types', axis=1)
y = df['encoded_story_types']

# Split data into training and test sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Split remaining data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# Create a transformer for preprocessing different types of input data
preprocessor = ColumnTransformer(
    transformers=[
        # Convert text data to TF-IDF features
        ('text1_tfidf', TfidfVectorizer(), 'story'),
        # Pass through numeric data unchanged
        ('num', 'passthrough', ['distracted', 'draining', 'openness', 'Concreteness_Story','emotion_score_norm',
        'num_sentences', 'avg_sentence_length', 'Sequentiality_FullSize', 'Topic_NLL_FullSize', 'fw_score_normalized'])
    ]
)

# Define hyperparameter grid for RandomForestClassifier
param_grid = {
    'randomforestclassifier__n_estimators': [100, 200, 300, 400, 500, 650, 800, 1000],
    'randomforestclassifier__max_depth': [10, 30, None],
    'randomforestclassifier__min_samples_split': [2, 5],
    'randomforestclassifier__min_samples_leaf': [1, 2]
}

# Define cross-validation values
cv_values = [3, 5, 7, 10]

# Iterate over different k-fold cross-validation values
for k in cv_values:
    print(f"\nPerforming grid search with {k}-fold cross-validation.\n")

    # Create a pipeline integrating the preprocessor and RandomForestClassifier
    pipeline = make_pipeline(
        preprocessor,
        RandomForestClassifier(random_state=42)
    )

    # Set up grid search with cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=k, scoring='accuracy', verbose=1)

    # Fit model to training data
    grid_search.fit(X_train, y_train)

    # Output best hyperparameters from grid search
    best_params = grid_search.best_params_
    print(f"Best parameters with {k}-fold CV: {best_params}")

    # Evaluate the model on the validation set
    val_predictions = grid_search.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    val_precision = precision_score(y_val, val_predictions, average='macro')
    val_recall = recall_score(y_val, val_predictions, average='macro')
    val_f1 = f1_score(y_val, val_predictions, average='macro')
    val_conf_matrix = confusion_matrix(y_val, val_predictions)

    # Output validation metrics
    print(f'\nValidation Metrics with {k}-fold CV:')
    print(f'Accuracy: {val_accuracy}')
    print(f'Precision: {val_precision}')
    print(f'Recall: {val_recall}')
    print(f'F1 Score: {val_f1}')
    print(f'Confusion Matrix:\n{val_conf_matrix}\n')

# Evaluate model on test data
test_predictions = grid_search.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
test_precision = precision_score(y_test, test_predictions, average='macro')
test_recall = recall_score(y_test, test_predictions, average='macro')
test_f1 = f1_score(y_test, test_predictions, average='macro')
test_conf_matrix = confusion_matrix(y_test, test_predictions)

# Output test metrics
print(f'\nTest Metrics:')
print(f'Accuracy: {test_accuracy}')
print(f'Precision: {test_precision}')
print(f'Recall: {test_recall}')
print(f'F1 Score: {test_f1}')
print(f'Confusion Matrix:\n{test_conf_matrix}')
