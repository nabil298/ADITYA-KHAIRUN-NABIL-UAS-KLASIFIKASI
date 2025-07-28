# Prediksi Tingkat Stres Berdasarkan Data Pengguna Baru

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Memuat Data
# Gantilah path dengan path file CSV yang sesuai jika dijalankan di lokal
url = '/content/drive/MyDrive/dataset/Student Mental Health Analysis During Online Learning.csv'
df = pd.read_csv(url)

# 2. Pra-pemrosesan Data
X = df[['Screen Time (hrs/day)', 'Sleep Duration (hrs)', 'Physical Activity (hrs/week)']]
y = df['Stress Level']

# Encode label target
le = LabelEncoder()
y = le.fit_transform(y)

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Grid Search untuk mencari n_neighbors terbaik
param_grid = {'n_neighbors': range(1, 31)}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_n = grid_search.best_params_['n_neighbors']
print(f"Nilai n_neighbors terbaik: {best_n}")

# 5. Pelatihan model terbaik
knn_best = KNeighborsClassifier(n_neighbors=best_n)
knn_best.fit(X_train, y_train)

# 6. Evaluasi
y_pred_best = knn_best.predict(X_test)
new_accuracy = accuracy_score(y_test, y_pred_best)
print(f"Akurasi Model Baru: {new_accuracy}")

if new_accuracy >= 0.6:
    print("Akurasi model telah mencapai target minimal 60%.")
else:
    print("Akurasi model masih di bawah 60%. Pertimbangkan untuk mencoba algoritma lain atau rekayasa fitur lebih lanjut.")

# 7. Prediksi berdasarkan input pengguna
print("\nMasukkan data baru untuk memprediksi tingkat stres:")

try:
    new_screen_time = float(input("Masukkan screen time (jam/hari): "))
    new_sleep_duration = float(input("Masukkan durasi tidur (jam): "))
    new_physical_activity = float(input("Masukkan aktivitas fisik (jam/minggu): "))

    # Buat DataFrame dari input user
    new_data_df = pd.DataFrame([[new_screen_time, new_sleep_duration, new_physical_activity]],
                                columns=['Screen Time (hrs/day)', 'Sleep Duration (hrs)', 'Physical Activity (hrs/week)'])

    # Prediksi
    predicted = knn_best.predict(new_data_df)
    predicted_label = le.inverse_transform(predicted)[0]

    print("\nUntuk data tersebut:")
    print(f"Prediksi Tingkat Stres: {predicted_label}")

except ValueError:
    print("Input tidak valid. Harap masukkan angka.")
except Exception as e:
    print(f"Terjadi kesalahan: {e}")
