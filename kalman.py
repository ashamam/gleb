import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Функция для фильтра Калмана
def kalman_filter(data, R, Q):
    n = len(data)
    x_est = np.zeros(n)  # оценка состояния
    P = np.zeros(n)  # оценка ошибки ковариации
    K = np.zeros(n)  # калмановский коэффициент

    x_est[0] = data[0]
    P[0] = 1.0

    for t in range(1, n):
        # Предсказание
        x_pred = x_est[t-1]
        P_pred = P[t-1] + Q

        # Обновление
        K[t] = P_pred / (P_pred + R)
        x_est[t] = x_pred + K[t] * (data[t] - x_pred)
        P[t] = (1 - K[t]) * P_pred

    return x_est

def evaluate_metrics(original, filtered):
    mse = mean_squared_error(original, filtered)
    mae = mean_absolute_error(original, filtered)
    r2 = r2_score(original, filtered)
    return mse, mae, r2

# Загрузка данных
df1 = pd.read_csv('./rightIris.csv')
df2 = pd.read_csv('./leftIris.csv')

df1 = df1.iloc[:3000:10]
df1 = df1[df1['coord'] >= 40]
df2 = df2.iloc[:3000:10]
df2 = df2[df2['coord'] >= 40]

# Параметры фильтра Калмана
R = 2  # шум измерения
Q = 15  # шум процесса

# Применение фильтра Калмана
start_time = time.time()
df1['pupil_size_filtered'] = kalman_filter(df1['coord'].values, R, Q)
kalman_time1 = time.time() - start_time
df2['pupil_size_filtered'] = kalman_filter(df2['coord'].values, R, Q)

mse_kalman1, mae_kalman1, r2_kalman1 = evaluate_metrics(df1['coord'], df1['pupil_size_filtered'])
mse_kalman2, mae_kalman2, r2_kalman2 = evaluate_metrics(df2['coord'], df2['pupil_size_filtered'])

print(kalman_time1)
print(f"Kalman Filter Data 1 - Time: {kalman_time1:.4f}s, MSE: {mse_kalman1:.4f}, MAE: {mae_kalman1:.4f}, R2: {r2_kalman1:.4f}")
print(f"Kalman Filter Data 2 - Time: {kalman_time1:.4f}s, MSE: {mse_kalman2:.4f}, MAE: {mae_kalman2:.4f}, R2: {r2_kalman2:.4f}")
# Построение первого графика (до и после применения фильтра)
plt.figure(figsize=(14, 8))
plt.plot(df1['time'], df1['coord'], label='Original Data 1', alpha=0.5)
plt.plot(df1['time'], df1['pupil_size_filtered'], label='Kalman Filtered Data 1', alpha=0.8)
plt.xlabel('Time')
plt.ylabel('Pupil Size')
plt.title('Original and Kalman Filtered Data 1')
plt.legend()


# Построение второго графика (до и после применения фильтра)
plt.figure(figsize=(14, 8))
plt.plot(df2['time'], df2['coord'], label='Original Data 2', alpha=0.5)
plt.plot(df2['time'], df2['pupil_size_filtered'], label='Kalman Filtered Data 2', alpha=0.8)
plt.xlabel('Time')
plt.ylabel('Pupil Size')
plt.title('Original and Kalman Filtered Data 2')
plt.legend()
plt.show()
