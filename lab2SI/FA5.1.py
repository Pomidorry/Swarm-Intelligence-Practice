import numpy as np

# Problem Definition
def f(x):
    g1=27 / (x[0] * x[1] ** 2 * x[2]) - 1>0
    g2=397.5 / (x[0] * x[1] ** 2 * x[2] ** 2) - 1>0
    g3=1.93 * x[3] ** 3 / (x[1] * x[2] * x[5] ** 4) - 1>0
    g4=1.93 / (x[1] * x[2] * x[6] ** 4) - 1>0
    g5=1.0 / (110 * x[5] ** 3) * np.sqrt((745.0 * x[3] / (x[1] * x[2])) ** 2 + 16.9 * 10 ** 6) - 1>0
    g6=1.0 / (85 * x[6] ** 3) * np.sqrt((745.0 * x[4] / (x[1] * x[2])) ** 2 + 157.5 * 10 ** 6) - 1>0
    g7=x[1] * x[2] / 40 - 1>0
    g8=1 * x[1] / x[0] - 1>0
    g9=x[0] / (12 * x[1]) - 1>0
    g10=(1.5 * x[5] + 1.9) / x[3] - 1>0
    g11=(1.1 * x[6] + 1.9) / x[4] - 1>0
    g12=x[0]<2.6
    g13=x[0]>3.6
    g14=x[1]<0.7
    g15=x[1]>0.8
    g16=np.round(x[2])<17
    g17=np.round(x[2])>28
    g18=x[3]<7.3
    g19=x[3]>8.3
    g20=x[4]<7.8
    g21=x[4]>8.3
    g22=x[5]<2.9
    g23=x[5]>3.9
    g24=x[6]<5.0
    g25=x[6]>5.5
    func=0.7854*x[0]*x[1]**2*(3.3333*np.round(x[2])**2 + 14.9334*np.round(x[2]) - 43.0934) - 1.508*x[0]*(x[5]**2 + x[6]**2) + 7.4777*(x[5]**3 + x[6]**3) + 0.7854*(x[3]*x[5]**2 + x[4]*x[6]**2)
    #func[g1 or g2 or g3 or g4 or g5 or g6 or g7 or g8 or g9 or g10 or g11]=10000
    if g1 or g2 or g3 or g4 or g5 or g6 or g7 or g8 or g9 or g10 or g11 or g12 or g13 or g14 or g15 or g16 or g17 or g18 or g19 or g20 or g21 or g22 or g23 or g24 or g25:
        func = 10000
    return func


nVar = 7
VarMin = np.array([2.6, 0.7, 17, 7.3, 7.8, 2.9, 5.0])
VarMax = np.array([3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5])

# Firefly Algorithm Parameters
MaxIt = 1000
nPop = 25
gamma = 1
beta0 = 2
alpha = 0.2
alpha_damp = 0.98
delta = 0.05 * (VarMax - VarMin)
m = 2

if isinstance(VarMin, (int, float)) and isinstance(VarMax, (int, float)):
    dmax = np.sqrt(nVar) * (VarMax - VarMin)
else:
    dmax = np.linalg.norm(VarMax - VarMin)

# Initialization
pop = np.empty(nPop, dtype=object)

BestSol = {"Position": None, "Cost": np.inf}

for i in range(nPop):
    pop[i] = {"Position": np.random.uniform(VarMin, VarMax), "Cost": np.inf}
    pop[i]["Cost"] = f(pop[i]["Position"])
    if pop[i]["Cost"] <= BestSol["Cost"]:
        BestSol = pop[i].copy()

BestCost = np.zeros(MaxIt)

# Firefly Algorithm Main Loop
for it in range(MaxIt):

    newpop = np.empty(nPop, dtype=object)

    for i in range(nPop):
        newpop[i] = {"Position": None, "Cost": np.inf}
        for j in range(nPop):
            if pop[j]["Cost"] < pop[i]["Cost"]:
                rij = np.linalg.norm(pop[i]["Position"] - pop[j]["Position"]) / dmax
                beta = beta0 * np.exp(-gamma * rij**m)
                e = delta * np.random.uniform(-1, 1, nVar)
                newsol = {"Position": None, "Cost": np.inf}

                newsol["Position"] = pop[i]["Position"] + beta * np.random.uniform(-1, 1, nVar) * (
                    pop[j]["Position"] - pop[i]["Position"]) + alpha * e
                newsol["Position"] = np.maximum(newsol["Position"], VarMin)
                newsol["Position"] = np.minimum(newsol["Position"], VarMax)

                newsol["Cost"] = f(newsol["Position"])

                if newsol["Cost"] <= newpop[i]["Cost"]:
                    newpop[i] = newsol.copy()
                    if newpop[i]["Cost"] <= BestSol["Cost"]:
                        BestSol = newpop[i].copy()

    pop = np.concatenate((pop, newpop))

    SortOrder = np.argsort([pop[i]["Cost"] for i in range(len(pop))])
    pop = pop[SortOrder]

    pop = pop[:nPop]

    BestCost[it] = BestSol["Cost"]

    print(f'Iteration {it}: Best Cost = {BestCost[it]}')

    alpha *= alpha_damp

# Results
import matplotlib.pyplot as plt
plt.figure()
plt.semilogy(BestCost, linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Best Cost')
plt.grid(True)
plt.show()
