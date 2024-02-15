# Finding feature importance using random forest

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif

df = pd.read_csv("hippo-final-data.csv")

original_feature_names = ['distracted', 'draining', 'openness', 'Concreteness_Story', 'num_sentences', 'avg_sentence_length','Sequentiality_FullSize', 'Topic_NLL_FullSize', 'emotion_score_norm', 'fw_score_normalized']

X = df[original_feature_names].values

y = df['encoded_story_types'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the classifier on all features
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Making the predictions
y_pred = rf.predict(X_test)

# Get and display feature importances
feature_importances = rf.feature_importances_

# DataFrame to display importances
importances_df = pd.DataFrame({
    'Feature': original_feature_names,
    'Importance': feature_importances
})

plt.figure(figsize=(10, 6))
plt.barh(importances_df['Feature'], importances_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.grid(axis='x', linestyle='--', linewidth=0.5)

plt.show()
