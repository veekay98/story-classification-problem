import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# Load data
df = pd.read_csv("hippo-final-data.csv")

# Separate dataset into features and target
X = df.drop('encoded_story_types', axis=1)
y = df['encoded_story_types']

# Divide data into training, validation, and test sets
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

# TF-IDF transformation of 'story' column
tfidf = TfidfVectorizer(max_features=100)
train_tfidf = tfidf.fit_transform(X_train['story']).toarray()
val_tfidf = tfidf.transform(X_val['story']).toarray()
test_tfidf = tfidf.transform(X_test['story']).toarray()

input_num_features = ['distracted', 'draining', 'openness', 'Concreteness_Story','emotion_score_norm',
                                                   'num_sentences', 'avg_sentence_length', 'Sequentiality_FullSize', 'Topic_NLL_FullSize', 'fw_score_normalized']

# Scale numeric features
scaler = StandardScaler()
num_features_train = scaler.fit_transform(X_train[input_num_features])
num_features_val = scaler.transform(X_val[input_num_features])
num_features_test = scaler.transform(X_test[input_num_features])

# Combine TF-IDF and numeric features
X_train_combined = np.hstack((train_tfidf, num_features_train))
X_val_combined = np.hstack((val_tfidf, num_features_val))
X_test_combined = np.hstack((test_tfidf, num_features_test))

# Reshape data for LSTM network input
X_train_combined = np.expand_dims(X_train_combined, axis=1)
X_val_combined = np.expand_dims(X_val_combined, axis=1)
X_test_combined = np.expand_dims(X_test_combined, axis=1)

# Building LSTM model
model = Sequential()
# The l2 regularization prevents overfitting
model.add(LSTM(128, input_shape=(1, X_train_combined.shape[2]), return_sequences=True, kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
# The dropout layer also helps prevent overfitting
model.add(Dropout(0.3))
model.add(LSTM(64, kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
# The sigmoid activation function is suitable for binary classification
model.add(Dense(1, activation='sigmoid'))

# Compile and train model
optimizer = Adam(learning_rate=0.0005)
# Binary cross entropy is suitable for binary classification
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(X_train_combined, y_train, epochs=20, batch_size=64)

# Evaluate model on validation set
val_predictions = (model.predict(X_val_combined) > 0.5).astype("int32") # Uses the threshold of 0.5 to set labels
val_accuracy = accuracy_score(y_val, val_predictions)
val_precision = precision_score(y_val, val_predictions)
val_recall = recall_score(y_val, val_predictions)
val_f1 = f1_score(y_val, val_predictions)
print(f'Validation Accuracy: {val_accuracy}')
print(f'Validation Precision: {val_precision}')
print(f'Validation Recall: {val_recall}')
print(f'Validation F1 Score: {val_f1}')

# Evaluate model on test set
test_predictions = (model.predict(X_test_combined) > 0.5).astype("int32") # Uses the threshold of 0.5 to set labels
test_accuracy = accuracy_score(y_test, test_predictions)
test_precision = precision_score(y_test, test_predictions)
test_recall = recall_score(y_test, test_predictions)
test_f1 = f1_score(y_test, test_predictions)
print(f'\nTest Metrics:')
print(f'Accuracy: {test_accuracy}')
print(f'Precision: {test_precision}')
print(f'Recall: {test_recall}')
print(f'F1 Score: {test_f1}')
