import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the data
df = pd.read_csv("hippo-final-data.csv")

# Separate the data into features and output
X = df.drop('encoded_story_types', axis=1)
y = df['encoded_story_types']

# Split the data into training/test and temporary data
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# Define a ColumnTransformer for preprocessing the features
preprocessor = ColumnTransformer(
    transformers=[
        # Transform 'story' column using TfidfVectorizer
        ('text1_tfidf', TfidfVectorizer(), 'story'),
        # Pass other numeric features directly
        ('num', 'passthrough', ['distracted', 'draining', 'openness', 'Concreteness_Story', 'emotion_score_norm',
                                'num_sentences', 'avg_sentence_length', 'Sequentiality_FullSize', 'Topic_NLL_FullSize',
                                'fw_score_normalized'])
    ]
)

# Define grid search parameters and k-fold cross-validation values
cv_values = [3, 5, 7, 10]
param_grid = {
    'svc__C': [1, 10, 100, 1000],
    'svc__gamma': ['scale', 'auto'],
    'svc__kernel': ['rbf', 'linear']
}

# Perform grid search for each k-fold value
for k in cv_values:
    print(f"\nPerforming grid search with {k}-fold cross-validation.\n")

    # Create a pipeline with TfidfVectorizer preprocessing and SVC classifier
    pipeline = make_pipeline(
        preprocessor,
        SVC(probability=True, random_state=42)
    )

    # Configure and perform the grid search
    grid_search = GridSearchCV(pipeline, param_grid, cv=k, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train)

    # Extract the best parameters from the grid search
    best_params = grid_search.best_params_
    print(f"Best parameters with {k}-fold CV: {best_params}")

    # Evaluate the model on the validation set
    val_predictions = grid_search.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    val_precision = precision_score(y_val, val_predictions, average='macro')
    val_recall = recall_score(y_val, val_predictions, average='macro')
    val_f1 = f1_score(y_val, val_predictions, average='macro')
    val_conf_matrix = confusion_matrix(y_val, val_predictions)

    # Print validation metrics
    print(f'\nValidation Metrics with {k}-fold CV:')
    print(f'Accuracy: {val_accuracy}')
    print(f'Precision: {val_precision}')
    print(f'Recall: {val_recall}')
    print(f'F1 Score: {val_f1}')
    print(f'Confusion Matrix:\n{val_conf_matrix}\n')

# Use the best model obtained from grid search to predict on the test set
best_model = grid_search.best_estimator_
test_predictions = best_model.predict(X_test)

# Calculate and display metrics for the test set
test_accuracy = accuracy_score(y_test, test_predictions)
test_precision = precision_score(y_test, test_predictions, average='macro')
test_recall = recall_score(y_test, test_predictions, average='macro')
test_f1 = f1_score(y_test, test_predictions, average='macro')
test_conf_matrix = confusion_matrix(y_test, test_predictions)

# Print the test metrics
print(f'\nTest Metrics:')
print(f'Accuracy: {test_accuracy}')
print(f'Precision: {test_precision}')
print(f'Recall: {test_recall}')
print(f'F1 Score: {test_f1}')
print(f'Confusion Matrix:\n{test_conf_matrix}')
