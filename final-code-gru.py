import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# Load data
df = pd.read_csv("hippo-final-data.csv")

# Separate features and output
X = df.drop('encoded_story_types', axis=1)
y = df['encoded_story_types']

# Splitting the data into training, validation, and test data
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# Converting text data in 'story' column to TF-IDF features
tfidf = TfidfVectorizer(max_features=100)
train_tfidf = tfidf.fit_transform(X_train['story']).toarray()
val_tfidf = tfidf.transform(X_val['story']).toarray()
test_tfidf = tfidf.transform(X_test['story']).toarray()

input_num_features = ['distracted', 'draining', 'openness', 'Concreteness_Story','emotion_score_norm',
                                                   'num_sentences', 'avg_sentence_length', 'Sequentiality_FullSize', 'Topic_NLL_FullSize', 'fw_score_normalized']

# Scaling numeric features to standardize them
scaler = StandardScaler()
num_features_train = scaler.fit_transform(X_train[input_num_features])
num_features_val = scaler.transform(X_val[input_num_features])
num_features_test = scaler.transform(X_test[input_num_features])

# Combining TF-IDF features with numeric features
X_train_combined = np.hstack((train_tfidf, num_features_train))
X_val_combined = np.hstack((val_tfidf, num_features_val))
X_test_combined = np.hstack((test_tfidf, num_features_test))

# Reshaping data for GRU input
X_train_combined = np.expand_dims(X_train_combined, axis=1)
X_val_combined = np.expand_dims(X_val_combined, axis=1)
X_test_combined = np.expand_dims(X_test_combined, axis=1)


# Building the model
model = Sequential()
# The l2 regularization prevents overfitting
model.add(GRU(128, input_shape=(1, X_train_combined.shape[2]), return_sequences=True, kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
# The dropout layer also helps prevent overfitting
model.add(Dropout(0.3))
model.add(GRU(64, kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
# The sigmoid activation function is suitable for binary classification
model.add(Dense(1, activation='sigmoid'))

# Compiling the model with Adam optimizer and binary cross-entropy loss function
optimizer = Adam(learning_rate=0.0005)
# Binary cross entropy is suitable for binary classification
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Training the model on the training set
model.fit(X_train_combined, y_train, epochs=20, batch_size=64)

# Evaluating the model's performance on the validation set
val_predictions = (model.predict(X_val_combined) > 0.5).astype("int32") # Setting threshold as 0.5
val_accuracy = np.mean(val_predictions.flatten() == y_val)
val_precision = precision_score(y_val, val_predictions)
val_recall = recall_score(y_val, val_predictions)
val_f1 = f1_score(y_val, val_predictions)

# Print validation metrics
print(f'Validation Accuracy: {val_accuracy}')
print(f'Validation Precision: {val_precision}')
print(f'Validation Recall: {val_recall}')
print(f'Validation F1 Score: {val_f1}')



# Predicting on the test set
test_predictions = (model.predict(X_test_combined) > 0.5).astype("int32")

# Calculate metrics for the test set
test_accuracy = accuracy_score(y_test, test_predictions)
test_precision = precision_score(y_test, test_predictions)
test_recall = recall_score(y_test, test_predictions)
test_f1 = f1_score(y_test, test_predictions)

# Print test metrics
print(f'\nTest Metrics:')
print(f'Accuracy: {test_accuracy}')
print(f'Precision: {test_precision}')
print(f'Recall: {test_recall}')
print(f'F1 Score: {test_f1}')
