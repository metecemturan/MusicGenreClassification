import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier, XGBRFClassifier
from sklearn.preprocessing import LabelEncoder

general_path = 'Data'
data = pd.read_csv(f'{general_path}/features_3_sec.csv')
data = data.iloc[0:, 1:]
data.head()

y = data['label'] 
X = data.loc[:, (data.columns != 'label') & (data.columns != 'length') & (data.columns != 'rolloff_var') & (data.columns != 'rolloff_mean')]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
print("Label mapping:", label_mapping)

#### NORMALIZE X ####
cols = X.columns
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X)

# New data frame with the new scaled data
X = pd.DataFrame(np_scaled, columns=cols)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model assessment function
def model_assess(model, title="Default"):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print('Accuracy', title, ':', round(accuracy_score(y_test, preds), 5), '\n')

# Cross Gradient Booster
xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05)
model_assess(xgb, "Cross Gradient Booster")

# Misclassification Analysis for Cross Gradient Booster
y_pred = xgb.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)

# Retrieve genre labels
genre_labels = label_encoder.classes_

# Calculate per-genre accuracy
genre_accuracies = {}
for i, genre in enumerate(genre_labels):
    correct_predictions = conf_matrix[i, i]
    total_actual = sum(conf_matrix[i, :])
    accuracy = correct_predictions / total_actual if total_actual > 0 else 0
    genre_accuracies[genre] = accuracy

# Initialize dictionary to store misclassification details
misclassification_counts = {genre: {other_genre: 0 for other_genre in genre_labels if other_genre != genre} for genre in genre_labels}

# Fill the misclassification details from the confusion matrix
for i, actual_genre in enumerate(genre_labels):
    for j, predicted_genre in enumerate(genre_labels):
        if i != j:  # Only consider misclassifications
            misclassification_counts[actual_genre][predicted_genre] = conf_matrix[i, j]

print("\nDetailed Misclassification Counts and Per-Genre Accuracy for Cross Gradient Booster:")
for actual_genre, misclassifications in misclassification_counts.items():
    print(f"Actual Genre: {actual_genre}")
    print(f"  Recall: {genre_accuracies[actual_genre]:.4f}")
    for predicted_genre, count in misclassifications.items():
        if count > 0:  # Only show misclassifications that occurred
            print(f"  Misclassified as {predicted_genre}: {count}")
    print()

# Converting the results to a DataFrame
misclassification_df = pd.DataFrame(misclassification_counts).T.fillna(0).astype(int)
accuracy_df = pd.DataFrame.from_dict(genre_accuracies, orient='index', columns=['Recall'])

# Combining the accuracy and misclassification into a single DataFrame
combined_df = misclassification_df.assign(Recall=accuracy_df['Recall'])
print("\nCombined DataFrame (with per-genre recall):")
print(combined_df)
