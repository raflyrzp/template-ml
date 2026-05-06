import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def linear_scaling(x):
    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)
    return (x - x_min) / (x_max - x_min)

def compute_mse(y, y_prime):
    N = len(y)
    return (1 / N) * np.sum((y - y_prime) ** 2)

def gradient_descent(x,y, w, b, alpha, epochs):
    N= len(y)
    for epoch in range(epochs):
        y_prime = np.dot(x, w) + b
        error = y_prime - y

        dw = (2/N) * np.dot(x.T, error)
        db = (2/N) * np.sum((error))

        w -= alpha * dw
        b -= alpha * db
    return w, b

df = pd.read_csv("online_vs_offline_learning_dataset.csv")
df_encoded = pd.get_dummies(df, columns=['Learning_Mode', 'Subject'], dtype=float)

X = df_encoded.drop('Exam_Score', axis=1).values.astype(float)
y = df_encoded['Exam_Score'].values.reshape(-1, 1).astype(float)

X_scaled = linear_scaling(X)

num_features = X_scaled.shape[1]
w = np.zeros((num_features, 1))
b = 0
alpha = 0.01
epochs = 1000

w_final, b_final = gradient_descent(X_scaled, y, w, b, alpha, epochs)

y_final_pred = np.dot(X_scaled, w_final) + b_final
mse = compute_mse(y, y_final_pred)
print(f"Mean Squared Error: {mse}")

res = pd.DataFrame({
    'Actual': y.flatten(),
    'Predicted': y_final_pred.flatten(),
    'Error': (y.flatten() - y_final_pred.flatten())
})

print("\n--- 10 Hasil Pertama ---")
print(res.head(10))

plt.figure(figsize=(10, 6))

# Scatter plot: Sumbu X adalah nilai asli, Sumbu Y adalah nilai prediksi
plt.scatter(y, y_final_pred, color='blue', alpha=0.5, label='Data Prediksi')

# Garis diagonal (Ideal: jika prediksi == asli, semua titik di garis ini)
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', lw=2, label='Garis Ideal (Perfect Fit)')

plt.title('Visualisasi Evaluasi: Nilai Aktual vs Prediksi')
plt.xlabel('Nilai Ujian Asli (y)')
plt.ylabel('Nilai Ujian Prediksi (y\')')
plt.legend()
plt.grid(True)
plt.show()
