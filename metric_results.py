from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd

general_path = 'Data'
data = pd.read_csv(f'{general_path}/features_3_sec.csv')
data = data.iloc[0:, 1:]

y = data['label']
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X = data.loc[:, (data.columns != 'label') & (data.columns != 'length') & (data.columns != 'rolloff_var') & (data.columns != 'rolloff_mean')]

# Normalizing
scaler = preprocessing.MinMaxScaler()
X = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train an XGBoost classifier
xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05)
xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)

overall_accuracy = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {overall_accuracy:.4f}")

# Generate a detailed classification report for per-genre metrics
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)

print("\nPer-Genre Metrics:")
genre_metrics = []

for genre, metrics in report.items():
    if genre in label_encoder.classes_:
        accuracy = metrics['f1-score']
        precision = metrics['precision']
        recall = metrics['recall']
        f1_score = metrics['f1-score']  # F1 score for each genre
        genre_metrics.append((genre, accuracy, precision, recall, f1_score))
        print(f"{genre}: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")
        
results_df = pd.DataFrame(genre_metrics, columns=['Genre', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
print("\nResults DataFrame:")
print(results_df)
