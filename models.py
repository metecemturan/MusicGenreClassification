from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import time
general_path = 'Data'

data = pd.read_csv(f'{general_path}/features_3_sec.csv')
data = data.iloc[0:, 1:]
data.head()

y = data['label']
X = data.loc[:, (data.columns != 'label') & (data.columns != 'length') & (data.columns != 'rolloff_var') & (data.columns != 'rolloff_mean') ]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

label_mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
print("Label mapping:", label_mapping)

#### NORMALIZE X ####
cols = X.columns
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X)

X = pd.DataFrame(np_scaled, columns=cols)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) 

def model_assess_with_time(model, title="Default"):
    start_time = time.time()  # Başlangıç zamanı
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    end_time = time.time()  # Bitiş zamanı

    accuracy = round(accuracy_score(y_test, preds), 5)
    elapsed_time = round(end_time - start_time, 5)  # Çalışma süresi
    print(f'Accuracy {title}: {accuracy}')
    print(f'{title} çalışma süresi: {elapsed_time} saniye\n')

# ---------------KNN-----------------
start_time = time.time()
param_grid = {'n_neighbors': [7, 9, 11, 13, 15]}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
end_time = time.time()

best_n_neighbors = grid_search.best_params_['n_neighbors']
print("Best parameters for KNN:", grid_search.best_params_)
print(f"KNN parametre araması çalışma süresi: {round(end_time - start_time, 5)} saniye\n")

knn = KNeighborsClassifier(n_neighbors=best_n_neighbors)
model_assess_with_time(knn, "KNN")

# ----------------SVM-----------------
param_grid = {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': [0.1, 1, 10, 100],
}

start_time = time.time()
svm = SVC(decision_function_shape="ovo")
grid_search = GridSearchCV(svm, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
end_time = time.time()

print("Best parameters found for SVM: ", grid_search.best_params_)
print(f"SVM parametre araması çalışma süresi: {round(end_time - start_time, 5)} saniye\n")

best_svm = grid_search.best_estimator_
model_assess_with_time(best_svm, "Best Support Vector Machine (with tuned parameters)")

# ----------------Logistic Regression------------------
lg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
model_assess_with_time(lg, "Logistic Regression")

# ----------------Random Forest------------------------
rforest = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0)
model_assess_with_time(rforest, "Random Forest")

# ----------------XGBoost Classifier--------------------
xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05)
model_assess_with_time(xgb, "Cross Gradient Booster")


