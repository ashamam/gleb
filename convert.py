import pandas as pd

# Укажите путь к вашему .txt файлу
input_file = './leftIris.txt'
output_file = './leftIris.csv'
columns = ['coord', 'time']

# Загрузка данных из .txt файла
# Здесь предполагается, что данные разделены табуляцией (\t)
data = pd.read_csv(input_file, delimiter=',', low_memory=False, header=None, names=columns)

# Сохранение данных в .csv файл
data.to_csv(output_file, index=False)
