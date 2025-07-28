mport pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Memuat Data
df = pd.read_csv('/content/drive/MyDrive/dataset/Student Mental Health Analysis During Online Learning.csv')

# 2. Pra-pemrosesan Data
# Menambahkan 'Physical Activity (hrs/week)' sebagai fitur
X = df[['Screen Time (hrs/day)', 'Sleep Duration (hrs)', 'Physical Activity (hrs/week)']]
y = df['Stress Level']

# Mengubah variabel target menjadi numerik
le = LabelEncoder()
y = le.fit_transform(y)

# 3. Pembagian Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Tuning Hiperparameter
# Mencari nilai n_neighbors terbaik
param_grid = {'n_neighbors': range(1, 31)}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_n = grid_search.best_params_['n_neighbors']
print(f"Nilai n_neighbors terbaik: {best_n}")

# 5. Pelatihan Model dengan n_neighbors terbaik
knn_best = KNeighborsClassifier(n_neighbors=best_n)
knn_best.fit(X_train, y_train)

# 6. Evaluasi Model
y_pred_best = knn_best.predict(X_test)
new_accuracy = accuracy_score(y_test, y_pred_best)

# 7. Menampilkan Hasil
print(f"Akurasi Model Baru: {new_accuracy}")

if new_accuracy >= 0.6:
    print("Akurasi model telah mencapai target minimal 60%.")
else:
    print("Akurasi model masih di bawah 60%. Pertimbangkan untuk mencoba algoritma lain atau rekayasa fitur lebih lanjut.")
