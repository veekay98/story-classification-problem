import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv("hippo-final-data.csv")

# Split the data into features and output
X = df.drop('encoded_story_types', axis=1)
y = df['encoded_story_types']

# Split dataset into a training/test set and a temporary set
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further split the temporary set into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# Define a ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        # Convert 'story' column to TF-IDF features
        ('text1_tfidf', TfidfVectorizer(), 'story'),
        # Pass numeric features directly
        ('num', 'passthrough', ['distracted', 'draining', 'openness', 'Concreteness_Story','emotion_score_norm',
       'num_sentences', 'avg_sentence_length', 'Sequentiality_FullSize', 'Topic_NLL_FullSize', 'fw_score_normalized'])
    ]
)

# Grid search parameter configuration for XGBClassifier
param_grid = {
    'xgbclassifier__n_estimators': [100, 200, 300],
    'xgbclassifier__learning_rate': [0.01, 0.1, 0.2],
    'xgbclassifier__max_depth': [3, 4, 5]
}

# Cross-validation values
cv_values = [3, 5, 7, 10]

# Iterating over different cross-validation values for grid search
for k in cv_values:
    print(f"\nPerforming grid search with {k}-fold cross-validation.\n")

    # Create a pipeline including the preprocessor and XGBClassifier
    pipeline = make_pipeline(
        preprocessor,
        XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    )

    # Set up grid search with cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=k, scoring='accuracy', verbose=1)

    # Fit the model using grid search on training data
    grid_search.fit(X_train, y_train)

    # Report best parameters from grid search
    best_params = grid_search.best_params_
    print(f"Best parameters with {k}-fold CV: {best_params}")

    # Evaluate model on validation set
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

# Retrieve the best model from the grid search
best_model = grid_search.best_estimator_

# Use the best model to predict on the test set
test_predictions = best_model.predict(X_test)

# Calculate and print test metrics
test_accuracy = accuracy_score(y_test, test_predictions)
test_precision = precision_score(y_test, test_predictions, average='macro')
test_recall = recall_score(y_test, test_predictions, average='macro')
test_f1 = f1_score(y_test, test_predictions, average='macro')
test_conf_matrix = confusion_matrix(y_test, test_predictions)

# Display test metrics
print(f'\nTest Metrics:')
print(f'Accuracy: {test_accuracy}')
print(f'Precision: {test_precision}')
print(f'Recall: {test_recall}')
print(f'F1 Score: {test_f1}')
print(f'Confusion Matrix:\n{test_conf_matrix}')
