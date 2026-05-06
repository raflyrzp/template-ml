import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def linear_scaling(X, X_min, X_max):
    selisih = X_max - X_min
    selisih[selisih == 0] = 1
    return (X - X_min) / selisih

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_log_loss(y_asli, y_prediksi):
    N = len(y_asli)
    epsilon = 1e-15
    y_prediksi = np.clip(y_prediksi, epsilon, 1 - epsilon)
    return -(1 / N) * np.sum(y_asli * np.log(y_prediksi) + (1 - y_asli) * np.log(1 - y_prediksi))

def gradient_descent_logistic(X, y, alpha, epochs):
    N = len(y)
    jumlah_fitur = X.shape[1]

    w = np.zeros((jumlah_fitur, 1))
    b = 0.0

    for i in range(epochs):
        z = np.dot(X, w) + b
        y_prime = sigmoid(z)

        error = y_prime - y

        dw = (1 / N) * np.dot(X.T, error)
        db = (1 / N) * np.sum(error)

        w = w - (alpha * dw)
        b = b - (alpha * db)

    return w, b

df = pd.read_csv("data_klasifikasi.csv")
target = "Target_Label"

df_encoded = pd.get_dummies(df, dtype=float)

X_df = df_encoded.drop(target, axis=1)
nama_fitur = X_df.columns.tolist()

X_raw = X_df.values.astype(float)
y = df_encoded[target].values.reshape(-1, 1).astype(float)

X_min = np.min(X_raw, axis=0)
X_max = np.max(X_raw, axis=0)
X_scaled = linear_scaling(X_raw, X_min, X_max)

print("Sedang melatih model...")
w_final, b_final = gradient_descent_logistic(X_scaled, y, alpha=0.1, epochs=2000)

y_prob = sigmoid(np.dot(X_scaled, w_final) + b_final)
log_loss = compute_log_loss(y, y_prob)

y_class = (y_prob >= 0.5).astype(int)
akurasi = np.mean(y_class == y) * 100

print("Training Selesai!")
print(f"Nilai Log Loss: {log_loss:.4f}")
print(f"Akurasi Model: {akurasi:.2f}%")

idx_sorted = np.argsort(y_prob.flatten())
plt.plot(y_prob.flatten()[idx_sorted], color='blue', label='Probabilitas (S-Curve)')
plt.scatter(range(len(y)), y.flatten()[idx_sorted], color='red', alpha=0.5, label='Kelas Asli (0 / 1)')
plt.axhline(0.5, color='green', linestyle='--', label='Threshold 0.5')
plt.xlabel('Data Index (Diurutkan)')
plt.ylabel('Probabilitas / Kelas')
plt.title('Regresi Logistik: Probabilitas vs Kelas Asli')
plt.legend()
plt.show()

print("\n--- TEBAK KELAS BARU ---")
print("Masukkan nilai (Ketik 1 atau 0 untuk fitur kategori):")

input_user = []
for fitur in nama_fitur:
    nilai = float(input(f"- {fitur}: "))
    input_user.append(nilai)

input_array = np.array(input_user).reshape(1, -1)
input_scaled = linear_scaling(input_array, X_min, X_max)

prob_tebakan = sigmoid(np.dot(input_scaled, w_final) + b_final)
kelas_tebakan = 1 if prob_tebakan[0][0] >= 0.5 else 0

print(f"\n=> Probabilitas: {prob_tebakan[0][0]:.4f}")
print(f"=> Prediksi Kelas {target}: {kelas_tebakan}")
