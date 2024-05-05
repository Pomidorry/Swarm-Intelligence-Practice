# differential evolution search of the two-dimensional sphere objective function
from numpy.random import rand
from numpy.random import choice
from numpy import asarray
from numpy import clip
from numpy import argmin
from numpy import min
from numpy import around
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf

def load_data():
    # Завантаження даних з xlsx-файлу
    global Data, xTrain, yTrain, xTest, yTest, x, y
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

load_data()

y_f=y.tolist()
x_f=x.tolist()
# define objective function
def obj(b):
    b1, b2 = b
    return sum([(y_f[i] - b1*(1-np.exp(-b2*x_f[i])))**2 for i in range(len(x_f))])

# define mutation operation
def mutation(x, F):
    return x[0] + F * (x[1] - x[2])


# define boundary check operation
def check_bounds(mutated, bounds):
    mutated_bound = [clip(mutated[i], bounds[i, 0], bounds[i, 1]) for i in range(len(bounds))]
    return mutated_bound


# define crossover operation
def crossover(mutated, target, dims, cr):
    # generate a uniform random value for every dimension
    p = rand(dims)
    # generate trial vector by binomial crossover
    trial = [mutated[i] if p[i] < cr else target[i] for i in range(dims)]
    return trial


def differential_evolution(pop_size, bounds, iter, F, cr):
    # initialise population of candidate solutions randomly within the specified bounds
    pop = bounds[:, 0] + (rand(pop_size, len(bounds)) * (bounds[:, 1] - bounds[:, 0]))
    # evaluate initial population of candidate solutions
    obj_all = [obj(ind) for ind in pop]
    # find the best performing vector of initial population
    best_vector = pop[argmin(obj_all)]
    best_obj = min(obj_all)
    prev_obj = best_obj
    # run iterations of the algorithm
    for i in range(iter):
        # iterate over all candidate solutions
        for j in range(pop_size):
            # choose three candidates, a, b and c, that are not the current one
            candidates = [candidate for candidate in range(pop_size) if candidate != j]
            a, b, c = pop[choice(candidates, 3, replace=False)]
            # perform mutation
            mutated = mutation([a, b, c], F)
            # check that lower and upper bounds are retained after mutation
            mutated = check_bounds(mutated, bounds)
            # perform crossover
            trial = crossover(mutated, pop[j], len(bounds), cr)
            # compute objective function value for target vector
            obj_target = obj(pop[j])
            # compute objective function value for trial vector
            obj_trial = obj(trial)
            # perform selection
            if obj_trial < obj_target:
                # replace the target vector with the trial vector
                pop[j] = trial
                # store the new objective function value
                obj_all[j] = obj_trial
        # find the best performing vector at each iteration
        best_obj = min(obj_all)
        # store the lowest objective function value
        if best_obj < prev_obj:
            best_vector = pop[argmin(obj_all)]
            prev_obj = best_obj
            # report progress at each iteration
            print('Iteration: %d f([%s]) = %.5f' % (i, around(best_vector, decimals=5), best_obj))
    return best_vector, best_obj



def model(b, x):
    # Кількість точок x та кількість параметрів b
    N = len(x)
    M = len(b)

    # Ініціалізуємо матрицю Y з нулями
    Y = np.zeros((N, M))
    print(b)
    # Обчислюємо Y для кожного параметра b
    Y = b[0] * (1 - np.exp(-b[1] * x))

    return Y

# def f(Y, y):
#     #M = len(Y)  # Кількість стовпців у матриці Y
#     N = len(y)     # Кількість елементів у векторі y

#     # Повторюємо вектор y, щоб він мав таку ж форму, як матриця Y
#     #y = np.tile(y, (1, M))
#     loss_list=[]
#     # Обчислюємо z
#     for i in range(2):
#         z = np.sqrt((Y[i] - y[i])**2) / N
#         loss_list.append(z)

#     return loss_list

def mse_loss(y_true, y_pred):
  """Calculates Mean Squared Error"""
  return (((np.array(y_true) - np.array(y_pred)) ** 2))/len(y_true)


def func(x, b1, b2):
    return b1 * (1 - np.exp(-b2 * x))

# define population size
pop_size = 100
# define lower and upper bounds for every dimension
bounds = asarray([(0, 100), (0, 1)])
# define number of iterations
iter = 100
# define scale factor for mutation
F = 0.5
# define crossover rate for recombination
cr = 0.7
 
# perform differential evolution
solution = differential_evolution(pop_size, bounds, iter, F, cr)
print(solution)

all_b=solution[0]
b_list=all_b.tolist()
b1, b2 = b_list[0], b_list[1]

x_values = np.linspace(0, max(x_f), 100)
y_values_func = func(x_values, b1, b2)

plt.scatter(x_f, y_f, label='Data')
plt.plot(x_values, y_values_func, color='red', label='Function: y = b1(1 - exp(-b2*x))')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Regression plot')
plt.legend()
plt.grid(True)
plt.show()

x_values_loss = np.linspace(0, max(x_f), 14)
y_values_func_loss = func(x_values_loss, b1, b2)

mse_losses=mse_loss(y_values_func_loss.tolist(), y_f)
plt.plot(y_values_func_loss, mse_losses, label='MSE')
plt.xlabel('Predicted Value (y_pred)')
plt.ylabel('Loss')
plt.show()
YModelTrain = model(b_list, xTrain)

# Получение предсказанных значений для тестового набора данных
YModelTest = model(b_list, xTest)

plt.plot(YModelTrain, yTrain, '.r', markersize=15)
plt.plot(np.arange(min(yTrain), max(yTrain), 0.03), np.arange(min(yTrain), max(yTrain), 0.03))
plt.xlabel('YModelTrain')
plt.ylabel('yTrain (Real Data)')
plt.grid(True)
#print(YModelTrain)
plt.axis([min(YModelTrain), max(YModelTrain), min(yTrain), max(yTrain)])
plt.show()
# График для тестовых данных

plt.plot(YModelTest, yTest, '.', markersize=15)
plt.plot(np.arange(min(yTest), max(yTest), 0.01), np.arange(min(yTest), max(yTest), 0.01))
plt.xlabel('YModelTest')
plt.ylabel('yTest (Real Data)')
plt.grid(True)
plt.axis([min(YModelTest), max(YModelTest), min(yTest), max(yTest)])
plt.show()


