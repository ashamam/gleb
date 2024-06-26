import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Функция для альфа-бета фильтра
def alpha_beta_filter(data, alpha=0.85, beta=0.005):
    n = len(data)
    x_est = np.zeros(n)
    x_pred = np.zeros(n)
    v_est = np.zeros(n)
    v_pred = np.zeros(n)
    
    x_est[0] = data[0]
    v_est[0] = 0  # начальная скорость
    
    for t in range(1, n):
        # Предсказание
        x_pred[t] = x_est[t-1] + v_est[t-1]
        v_pred[t] = v_est[t-1]
        
        # Обновление
        residual = data[t] - x_pred[t]
        x_est[t] = x_pred[t] + alpha * residual
        v_est[t] = v_pred[t] + beta * residual
        
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

# Применение альфа-бета фильтра
alpha = 0.85  # Параметр сглаживания
beta = 0.005  # Параметр коррекции скорости


start_time = time.time()
df1['pupil_size_filtered'] = alpha_beta_filter(df1['coord'].values, alpha, beta)
time1 = time.time() - start_time
df2['pupil_size_filtered'] = alpha_beta_filter(df2['coord'].values, alpha, beta)

mse, mae, r2 = evaluate_metrics(df1['coord'], df1['pupil_size_filtered'])
mse2, mae2, r2_2 = evaluate_metrics(df2['coord'], df2['pupil_size_filtered'])

print(time1)
print(f"a/b Filter Data 1 - Time: {time1:.4f}s, MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
print(f"a/b Filter Data 2 - Time: {time1:.4f}s, MSE: {mse2:.4f}, MAE: {mae2:.4f}, R2: {r2_2:.4f}")

# Построение первого графика (до и после применения фильтра)
plt.figure(figsize=(14, 8))
plt.plot(df1['time'], df1['coord'], label='Original Data 1', alpha=0.5)
plt.plot(df1['time'], df1['pupil_size_filtered'], label='Alpha-Beta Filtered Data 1', alpha=0.8)
plt.xlabel('Time')
plt.ylabel('Pupil Size')
plt.title('Original and Alpha-Beta Filtered Data 1')
plt.legend()


# Построение второго графика (до и после применения фильтра)
plt.figure(figsize=(14, 8))
plt.plot(df2['time'], df2['coord'], label='Original Data 2', alpha=0.5)
plt.plot(df2['time'], df2['pupil_size_filtered'], label='Alpha-Beta Filtered Data 2', alpha=0.8)
plt.xlabel('Time')
plt.ylabel('Pupil Size')
plt.title('Original and Alpha-Beta Filtered Data 2')
plt.legend()
plt.show()
