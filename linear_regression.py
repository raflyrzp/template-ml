import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def linear_scaling(X, X_min, X_max):
    return (X - X_min) / (X_max - X_min)

def compute_mse(y_asli, y_prediksi):
    N = len(y_asli)
    return (1 / N) * np.sum((y_asli - y_prediksi)**2)

def gradient_descent(X, y, alpha, epochs):
    N = len(y)
    jumlah_fitur = X.shape[1]

    w = np.zeros((jumlah_fitur, 1))
    b = 0.0

    for i in range(epochs):
        y_prime = np.dot(X, w) + b
        error = y_prime - y

        dw = (2 / N) * np.dot(X.T, error)
        db = (2 / N) * np.sum(error)

        w = w - (alpha * dw)
        b = b - (alpha * db)

    return w, b

df = pd.read_csv("data_ujian.csv")
target = "Exam_Score"

X_raw = df.drop(target, axis=1).values
y = df[target].values.reshape(-1, 1)

X_min = np.min(X_raw, axis=0)
X_max = np.max(X_raw, axis=0)
X_scaled = linear_scaling(X_raw, X_min, X_max)

print("Sedang melatih model...")
w_final, b_final = gradient_descent(X_scaled, y, alpha=0.01, epochs=1000)

y_final_pred = np.dot(X_scaled, w_final) + b_final
mse = compute_mse(y, y_final_pred)
print(f"Training Selesai! Nilai MSE: {mse:.4f}")

plt.scatter(y, y_final_pred, color='blue', label='Prediksi')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', label='Ideal')
plt.xlabel('Nilai Asli')
plt.ylabel('Nilai Tebakan')
plt.title('Regresi Linear: Asli vs Prediksi')
plt.legend()
plt.show()

print("\n--- TEBAK NILAI BARU ---")
jumlah_fitur = X_raw.shape[1]
print(f"Masukkan {jumlah_fitur} angka dipisah dengan spasi.")
input_user = input("Input angka: ")

angka_list = [float(i) for i in input_user.split()]
input_array = np.array(angka_list).reshape(1, -1)

input_scaled = linear_scaling(input_array, X_min, X_max)

hasil_tebakan = np.dot(input_scaled, w_final) + b_final
print(f"\n=> Prediksi {target}: {hasil_tebakan[0][0]:.2f}")
