import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Функция для медианного фильтра
def apply_median_filter(data, window_size=5):
    return median_filter(data, size=window_size)

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

# Применение медианного фильтра
window_size = 5  # Вы можете настроить размер окна

start_time = time.time()
df1['pupil_size_filtered'] = apply_median_filter(df1['coord'].values, window_size)
time1 = time.time() - start_time
df2['pupil_size_filtered'] = apply_median_filter(df2['coord'].values, window_size)

mse, mae, r2 = evaluate_metrics(df1['coord'], df1['pupil_size_filtered'])
mse2, mae2, r2_2 = evaluate_metrics(df2['coord'], df2['pupil_size_filtered'])

print(time1)
print(f"median Filter Data 1 - Time: {time1:.4f}s, MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
print(f"median Filter Data 2 - Time: {time1:.4f}s, MSE: {mse2:.4f}, MAE: {mae2:.4f}, R2: {r2_2:.4f}")

# Построение первого графика (до и после применения фильтра)
plt.figure(figsize=(14, 8))
plt.plot(df1['time'], df1['coord'], label='Original Data 1', alpha=0.5)
plt.plot(df1['time'], df1['pupil_size_filtered'], label='Filtered Data 1', alpha=0.8)
plt.xlabel('Time')
plt.ylabel('Pupil Size')
plt.title('Original and Median Filtered Data 1')
plt.legend()


# Построение второго графика (до и после применения фильтра)
plt.figure(figsize=(14, 8))
plt.plot(df2['time'], df2['coord'], label='Original Data 2', alpha=0.5)
plt.plot(df2['time'], df2['pupil_size_filtered'], label='Filtered Data 2', alpha=0.8)
plt.xlabel('Time')
plt.ylabel('Pupil Size')
plt.title('Original and Median Filtered Data 2')
plt.legend()
plt.show()
