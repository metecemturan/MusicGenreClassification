from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from xgboost import XGBClassifier

general_path = 'Data'
data = pd.read_csv(f'{general_path}/features_3_sec.csv')
data = data.iloc[0:, 1:]

y = data['label']
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X = data.loc[:, (data.columns != 'label') & (data.columns != 'length') & (data.columns != 'rolloff-')]

# Normalizing 
scaler = preprocessing.MinMaxScaler()
X = scaler.fit_transform(X)

# Spliting the data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Training an XGBoost classifier
xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05)
xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)

overall_accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {overall_accuracy:.4f}")

# Displaying the confusion matrix as a heatmap
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

class_accuracies = cm.diagonal() / cm.sum(axis=1)
for genre, accuracy in zip(label_encoder.classes_, class_accuracies):
    print(f"{genre} accuracy: {accuracy:.4f}")

overall_accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {overall_accuracy:.4f}")
