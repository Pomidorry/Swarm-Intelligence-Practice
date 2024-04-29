# differential evolution search of the two-dimensional sphere objective function
from numpy.random import rand
from numpy.random import choice
from numpy import asarray
from numpy import clip
from numpy import argmin
from numpy import min
from numpy import around
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    # Завантаження даних з xlsx-файлу
    global Data, xTrain, yTrain, xTest, yTest, x, y
    Data = pd.read_excel("E:\\SI\\lab6SI\\DataRegression.xlsx", sheet_name="Var09")
    
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



def subtract_lists(list1, list2):
    return [item for item in list1 if item not in list2]

def lossfunc(Y, y):
    M = len(Y)  # Кількість стовпців у матриці Y
    N = len(y)     # Кількість елементів у векторі y

    # Повторюємо вектор y, щоб він мав таку ж форму, як матриця Y

    # Обчислюємо z
    Y1=sorted(Y)
    Y1=np.array(Y1)
    y1=sorted(y)
    y1=np.array(y1)
    z = np.sqrt((Y1 - y1)**2 / N)

    return z


def visualization():
    global Data, b_best, xTrain, yTrain, xTest, yTest, MaxIter, GlobalBestFObj, LossFuncFObjInTest

    # Отримання мінімального та максимального значення x
    xMin = np.min(Data.iloc[:, 1])
    xMax = np.max(Data.iloc[:, 1])

    # Кількість точок для візуалізації
    N = len(Data.iloc[:, 1])
    h = (xMax - xMin) / (N - 1)
    X = np.linspace(xMin, xMax, N)

    # Обчислення Y за найкращими параметрами моделі
    Y = model(b_best, X)
    print(Y)

    # Візуалізація
    plt.figure(1)
    plt.plot(Data.iloc[:, 1], Data.iloc[:, 0], '.', X, Y)
    plt.grid(True)
    plt.xlabel('X')
    plt.ylabel('Y (Real, Model)')
    plt.legend(['Real Data Y', 'Model Data Y'])

    plt.figure(2)

# Задання кількості ітерацій
    it = np.arange(1, MaxIter)[::3]
    y=Data.iloc[:, 0]
    print(y.tolist())
    LossFuncFObjInTest = lossfunc(Y, y.tolist())
    print(LossFuncFObjInTest)
    # Побудова підграфіків
    # plt.subplot(2, 2, 1)
    # plt.plot(it, GlobalBestFObj)
    # plt.grid(True)
    # plt.xlabel('Iterations')
    # plt.ylabel('GlobalBestFObj')

    plt.subplot(2, 2, 2)
    plt.plot(it, LossFuncFObjInTest)
    plt.grid(True)
    plt.xlabel('Iterations')
    plt.ylabel('LossFuncFObjInTest')

    # plt.subplot(2, 1, 2)
    # plt.plot(it, GlobalBestFObj, it, LossFuncFObjInTest)
    # plt.grid(True)
    # plt.xlabel('Iterations')
    # plt.ylabel('FuncFObj (Train, Test)')
    # plt.legend(['Train', 'Test'])
    plt.show()



# define objective function
def obj(x, y, b_pop):
    J = sum(np.power((y - (b_pop[0] + b_pop[1]*np.exp((-1)*x*b_pop[3])+b_pop[2]*np.exp((-1)*x*b_pop[4]))), 2))
    return J, b_pop[0], b_pop[1]

#y=b1(1-e^(-b2*x))
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


def differential_evolution(xTrain, yTrain, pop_size, pop, bounds, iter, F, cr):
    # initialise population of candidate solutions randomly within the specified bounds
    
    # evaluate initial population of candidate solutions
    all_values = [obj(xTrain, yTrain, ind) for ind in pop]
    #print(objective_values)
    # find the best performing vector of initial population
    objective_values = [item[0] for item in all_values]
    #print(objective_values)
    best_obj = min(objective_values)
    #print(best_obj)
    #print(objective_values[:][:][0])
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
            obj_target = obj(xTrain, yTrain, pop[j])
            #print(obj_target)
            #obj_target = [item[0] for item in obj_target1]
            # compute objective function value for trial vector
            obj_trial = obj(xTrain, yTrain, trial)
            #obj_trial = [item[0] for item in obj_trial1]
            # perform selection
            if obj_trial[0] < obj_target[0]:
                # replace the target vector with the trial vector
                pop[j] = trial
                # store the new objective function value
                objective_values[j] = obj_trial[0]
        # find the best performing vector at each iteration
        #print(objective_values)
        best_obj = min(objective_values)
        #print(best_obj)
        # store the lowest objective function value
        if best_obj < prev_obj:
            prev_obj = best_obj
            # report progress at each iteration
            print('Iteration: %d f([%s]) = %.5f' % (i, around(pop[argmin(objective_values)], decimals=5), best_obj))
    return [pop[argmin(objective_values)], best_obj]

global Data, xTrain, yTrain, xTest, yTest, x, y

# define population size
pop_size = 150
# define lower and upper bounds for every dimension
# define number of iterations
MaxIter = 100
# define scale factor for mutation
F = 0.5
# define crossover rate for recombination
cr = 0.7
load_data()

x_arr = np.array(x)
y_arr = np.array(y)
bounds = asarray([(0, 10), (0, 5), (-2, 2), (-2, 2), (-2, 2)])
#print(xTrain)
# perform differential evolution
pop = bounds[:, 0] + (rand(pop_size, len(bounds)) * (bounds[:, 1] - bounds[:, 0]))

solution = differential_evolution(x_arr, y_arr, pop_size, pop, bounds, MaxIter, F, cr)
print(solution)


def model(b, x):
    # Кількість точок x та кількість параметрів b
    N = len(x)
    M = len(b)
    # Ініціалізуємо матрицю Y з нулями
    Y = np.zeros((N, M))
    # Обчислюємо Y для кожного параметра b
    for i in range(M):
        Y[:,i] = b[0] + b[1]*np.exp(-x*b[3])+b[2]*np.exp(-x*b[4])
    return Y[:, 0]
tem=solution[0]
print(tem)
b_best = tem.tolist()
print(b_best)
visualization()