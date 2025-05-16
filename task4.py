import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
#імпортуємо потрібні бібліотеки

url = "https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/data/renfe_small.csv"
df = pd.read_csv(url)
# Завантажуємо дані з URL

df = df.dropna()
# Видаляємо рядки з пропущеними значеннями

df['start_date'] = pd.to_datetime(df['start_date'])
df['end_date'] = pd.to_datetime(df['end_date'])
# Перетворюємо стовпці дат у формат datetime

df['duration'] = (df['end_date'] - df['start_date']).dt.total_seconds() / 60
# Обчислюємо тривалість поїздки в хвилинах

df['cheap'] = (df['price'] < 50).astype(int)
# Створюємо новий стовпець 'cheap', який вказує, чи є поїздка дешевою (ціна менше 50)


cat_cols = ['train_type', 'origin', 'destination', 'train_class', 'fare']
encoders = {col: LabelEncoder().fit(df[col]) for col in cat_cols}
for col in cat_cols:
    df[col] = encoders[col].transform(df[col])
# Кодуємо категоріальні дані та замінюємо їх закодованими значеннями

X = df[['train_type', 'origin', 'destination', 'train_class', 'fare', 'duration']]
y = df['cheap']
# Розділяємо дані на вхідні (X) та вихідні (y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Розділяємо дані на навчальну та тестову вибірки в співвідношенні 70:30

model = GaussianNB()
model.fit(X_train, y_train)
# Створюємо та навчаємо модель

y_pred = model.predict(X_test)
# Прогнозуємо результати для тестової вибірки

accuracy = accuracy_score(y_test, y_pred)
print("Точність моделі:", round(accuracy * 100, 2), "%")
# Виводимо оцінку точність моделі у консоль