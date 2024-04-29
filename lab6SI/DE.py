import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    # Завантаження даних з xlsx-файлу
    global Data, xTrain, yTrain, xTest, yTest
    Data = pd.read_excel("E:\\SI\\lab6SI\\DataRegression.xlsx", sheet_name="Var01")
    
    # Перемішування рядків з даними, що отримані з xlsx-файлу
    Data = Data.sample(frac=1).reset_index(drop=True)
    
    # Розділення на ознаки (x) і цільову змінну (y)
    x = Data.iloc[:, 1]  # Ознака - другий стовпець
    y = Data.iloc[:, 0]  # Цільова змінна - перший стовпець
    
    # Розділення на навчальну та тестову вибірки
    TrainPercent = 75
    LenDataTrain = round(len(Data) * TrainPercent / 100)
    xTrain, xTest = x[:LenDataTrain], x[LenDataTrain:]
    yTrain, yTest = y[:LenDataTrain], y[LenDataTrain:]


def model(b, x):
    # Кількість точок x та кількість параметрів b
    N = len(x)
    M = len(b)

    # Ініціалізуємо матрицю Y з нулями
    Y = np.zeros((N, M))

    # Обчислюємо Y для кожного параметра b
    for i in range(M):
        Y[:, i] = b[i, 0] * (1 - np.exp(-b[i, 1] * x))

    return Y

def f(Y, y):
    M = len(Y[0])  # Кількість стовпців у матриці Y
    N = len(y)     # Кількість елементів у векторі y

    # Повторюємо вектор y, щоб він мав таку ж форму, як матриця Y
    y = np.tile(y, (1, M))

    # Обчислюємо z
    z = np.sqrt(np.sum((Y - y)**2) / N)

    return z

def visualization():
    global Data, b_best, xTrain, yTrain, xTest, yTest, MaxIter, GlobalBestFObj, LossFuncFObjInTest

    # Отримання мінімального та максимального значення x
    xMin = np.min(Data.iloc[:, 1])
    xMax = np.max(Data.iloc[:, 1])

    # Кількість точок для візуалізації
    N = 345
    h = (xMax - xMin) / (N - 1)
    X = np.linspace(xMin, xMax, N)

    # Обчислення Y за найкращими параметрами моделі
    Y = model(b_best, X)

    # Візуалізація
    plt.figure(1)
    plt.plot(Data.iloc[:, 1], Data.iloc[:, 0], '.', X, Y)
    plt.grid(True)
    plt.xlabel('X')
    plt.ylabel('Y (Real, Model)')
    plt.legend(['Real Data Y', 'Model Data Y'])
    plt.show()


load_data()
print(xTrain)