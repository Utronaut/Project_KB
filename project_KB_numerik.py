import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR

# Simulasi dataset (ganti dengan dataset Anda)
np.random.seed(42)
data_size = 500
data = {
    'hour_of_day': np.random.randint(0, 24, data_size),  # Jam
    'day_of_week': np.random.randint(0, 7, data_size),   # Hari
    'vehicles_count': np.random.randint(50, 500, data_size),
    'traffic_level': np.random.randint(0, 100, data_size),  # Tingkat kemacetan
}

# Membuat DataFrame
df = pd.DataFrame(data)

# Membagi data menjadi fitur dan label
X = df[['hour_of_day', 'day_of_week', 'vehicles_count']]
y = df['traffic_level']

# Split data untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: KNN
knn = KNeighborsRegressor()
knn_params = {'n_neighbors': [3, 5, 7, 9]}
gs_knn = GridSearchCV(knn, knn_params, cv=5, scoring='neg_mean_squared_error')
gs_knn.fit(X_train, y_train)
knn_best = gs_knn.best_estimator_
knn_pred = knn_best.predict(X_test)
knn_mse = mean_squared_error(y_test, knn_pred)

# Model 2: Naive Bayes
# Menggunakan GaussianNB (sesuai untuk data kontinu tetapi kurang optimal untuk regresi)
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
nb_mse = mean_squared_error(y_test, nb_pred)

# Model 3: SVR
svr = SVR()
svr_params = {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10], 'epsilon': [0.1, 0.2]}
gs_svr = GridSearchCV(svr, svr_params, cv=5, scoring='neg_mean_squared_error')
gs_svr.fit(X_train, y_train)
svr_best = gs_svr.best_estimator_
svr_pred = svr_best.predict(X_test)
svr_mse = mean_squared_error(y_test, svr_pred)

# Membandingkan hasil
print("Hasil MSE masing-masing algoritma:")
print(f"KNN: {knn_mse}")
print(f"Naive Bayes: {nb_mse}")
print(f"SVR: {svr_mse}")

# Menentukan algoritma terbaik
mse_results = {"KNN": knn_mse, "Naive Bayes": nb_mse, "SVR": svr_mse}
best_model = min(mse_results, key=mse_results.get)
print(f"Algoritma terbaik berdasarkan MSE adalah {best_model} dengan MSE: {mse_results[best_model]}")