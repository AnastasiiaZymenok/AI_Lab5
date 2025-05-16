import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import LabelEncoder
#Іпортуємо біліотеки

data = pd.DataFrame([
    ['Sunny', 'High', 'Weak', 'No'],
    ['Sunny', 'High', 'Strong', 'No'],
    ['Overcast', 'High', 'Weak', 'Yes'],
    ['Rain', 'High', 'Weak', 'Yes'],
    ['Rain', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Normal', 'Strong', 'No'],
    ['Overcast', 'Normal', 'Strong', 'Yes'],
    ['Sunny', 'High', 'Weak', 'No'],
    ['Sunny', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Normal', 'Weak', 'Yes'],
    ['Sunny', 'Normal', 'Strong', 'Yes'],
    ['Overcast', 'High', 'Strong', 'Yes'],
    ['Overcast', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'High', 'Strong', 'No']
], columns=pd.Index(['Outlook', 'Humidity', 'Wind', 'Play']))
# Створюємо датафрейм з потрібними даними

encoders = {}
for col in data.columns:
    encoders[col] = LabelEncoder()
    data[col] = encoders[col].fit_transform(data[col])
# Кодуємо категоріальні дані. Для кожного стовпця створюємо енкодер

X = data[['Outlook', 'Humidity', 'Wind']]
y = data['Play']
# Розділяємо дані на вхідні (X) та вихідні (y)

model = CategoricalNB()
model.fit(X, y)
# Створюємо та навчаємо модель

input_values = {
    'Outlook': 'Overcast',
    'Humidity': 'High',
    'Wind': 'Strong'
}
# Вхідні дані для прогнозу

test = pd.DataFrame([[
    encoders['Outlook'].transform([input_values['Outlook']])[0],
    encoders['Humidity'].transform([input_values['Humidity']])[0],
    encoders['Wind'].transform([input_values['Wind']])[0],
]], columns=pd.Index(['Outlook', 'Humidity', 'Wind']))
# кодуємо тестові дані та створюємо датафрейм

pred = model.predict(test)
proba = model.predict_proba(test)
#Прогнозуємо результат та ймовірності

print("Чи буде матч:", encoders['Play'].inverse_transform(pred)[0])
print("Ймовірності (No/Yes):", proba)
#Виводимо результат у консоль