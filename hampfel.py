import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import ParameterGrid
from scipy.optimize import curve_fit
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score




def hampel_filter(data, window_size, n_sigma):
    """
    Фильтр Хампеля для удаления выбросов из данных.

    Параметры:
        - data: массив данных
        - window_size: размер окна для рассмотрения
        - n_sigma: количество стандартных отклонений для определения выбросов

    Возвращает:
        - filtered_data: массив данных после применения фильтра
    """
    data = data.copy()
    n = len(data)
    k = 1.4826  # Константа для нормального распределения

    filtered_data = data.copy()
    for i in range(window_size // 2, n - window_size // 2):
        window = data[i - window_size // 2:i + window_size // 2 + 1]
        med = np.median(window)
        mad = k * np.median(np.abs(window - med))
        threshold = n_sigma * mad
        if np.abs(data[i] - med) > threshold:
            filtered_data[i] = med

    return filtered_data

def evaluate_metrics(original, filtered):
    mse = mean_squared_error(original, filtered)
    mae = mean_absolute_error(original, filtered)
    r2 = r2_score(original, filtered)
    return mse, mae, r2

window_size = 5
n_sigma = 3
df = pd.read_csv('./rightIris.csv')
df2 = pd.read_csv('./leftIris.csv')
columns = ['coord', 'time']

#print(df.head())
df1 = df.iloc[:3000:10]
df1 = df1[df1['coord'] >= 40]
df2 = df2.iloc[:3000:10]
df2 = df2[df2['coord'] >= 40]
array = df1.iloc[:, 0].values

#df1['pupil_size_filtered'] = hampel_filter(df1['coord'].values)


#plt.figure(figsize=(14, 8))

# for window_size in window_sizes:
#     for n_sigma in n_sigmas:

start_time = time.time()
df1['pupil_size_filtered'] = hampel_filter(df1['coord'].values, window_size, n_sigma)
time1 = time.time() - start_time
df2['pupil_size_filtered'] = hampel_filter(df2['coord'].values, 120, 3)
        
        # Построение графика отфильтрованных данных
        #plt.plot(df1['time'], df1['pupil_size_filtered'], label=f'Window {window_size}, Sigma {n_sigma}', alpha=0.6)

# Исходные данные
#plt.plot(df1['time'], df1['coord'], label='Original Pupil Size', alpha=0.3, color='black')
df1.dropna(subset=['coord', 'pupil_size_filtered'], inplace=True)
df2.dropna(subset=['coord', 'pupil_size_filtered'], inplace=True)
mse, mae, r2 = evaluate_metrics(df1['coord'], df1['pupil_size_filtered'])
mse2, mae2, r2_2 = evaluate_metrics(df2['coord'], df2['pupil_size_filtered'])

print(time1)
print(f"hampfel Filter Data 1 - Time: {time1:.4f}s, MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
print(f"hampfel Filter Data 2 - Time: {time1:.4f}s, MSE: {mse2:.4f}, MAE: {mae2:.4f}, R2: {r2_2:.4f}")

# param_grid = {
#     'window_size': [7, 15, 30, 60],
#     'n_sigma': [2, 3, 4, 5]
# }

# best_score = float('inf')
# best_params = None
# best_filtered_data = None

# for params in ParameterGrid(param_grid):
#     filtered_data = hampel_filter(df1['coord'].values, params['window_size'], params['n_sigma'])
    
#     # Оценка качества фильтрации, например, по сумме квадратов отклонений
#     score = np.sum((df1['coord'].values - filtered_data) ** 2)
    
#     if score < best_score:
#         best_score = score
#         best_params = params
#         best_filtered_data = filtered_data



# Построение графика с лучшими параметрами
plt.figure(figsize=(12, 6))
plt.plot(df1['time'], df1['coord'], label='Original Pupil Size', alpha=0.5, marker='o', linestyle='-')
plt.plot(df1['time'], df1['pupil_size_filtered'], label=f"Filtered (window={30}, sigma={4})", alpha=0.8, marker='o', linestyle='-')

plt.title('Pupil Size Over Time with Optimized Hampel Filter')
plt.xlabel('Time')
plt.ylabel('Pupil Size')
plt.legend()
plt.grid(True)
#plt.show()

plt.figure(figsize=(12, 6))

plt.plot(df2['time'], df2['coord'], label='Original Pupil Size', alpha=0.5, marker='o', linestyle='-')
plt.plot(df2['time'], df2['pupil_size_filtered'], label=f"Filtered (window=30, sigma=4)", alpha=0.8, marker='o', linestyle='-')

plt.title('Pupil Size Over Time with Optimized Hampel Filter')
plt.xlabel('Time')
plt.ylabel('Pupil Size')
plt.legend()
plt.grid(True)
plt.show()