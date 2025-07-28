import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

st.title("Prediksi Tingkat Stres Mahasiswa")

# Upload CSV
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Pra-pemrosesan
    X = df[['Screen Time (hrs/day)', 'Sleep Duration (hrs)', 'Physical Activity (hrs/week)']]
    y = df['Stress Level']
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Grid Search
    param_grid = {'n_neighbors': range(1, 31)}
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_n = grid_search.best_params_['n_neighbors']

    st.success(f"Nilai n_neighbors terbaik: {best_n}")

    # Pelatihan model
    knn_best = KNeighborsClassifier(n_neighbors=best_n)
    knn_best.fit(X_train, y_train)

    # Evaluasi
    y_pred_best = knn_best.predict(X_test)
    acc = accuracy_score(y_test, y_pred_best)
    st.info(f"Akurasi Model: {acc:.2f}")

    # Form input pengguna
    st.header("Prediksi Tingkat Stres Berdasarkan Data Baru")
    screen_time = st.number_input("Screen Time (jam/hari)", min_value=0.0, value=4.0)
    sleep_duration = st.number_input("Durasi Tidur (jam)", min_value=0.0, value=6.0)
    physical_activity = st.number_input("Aktivitas Fisik (jam/minggu)", min_value=0.0, value=3.0)

    if st.button("Prediksi"):
        new_data = pd.DataFrame([[screen_time, sleep_duration, physical_activity]],
                                columns=['Screen Time (hrs/day)', 'Sleep Duration (hrs)', 'Physical Activity (hrs/week)'])
        prediction = knn_best.predict(new_data)
        pred_label = le.inverse_transform(prediction)[0]
        st.success(f"Prediksi Tingkat Stres: {pred_label}")
else:
    st.warning("Silakan unggah file dataset CSV terlebih dahulu.")
