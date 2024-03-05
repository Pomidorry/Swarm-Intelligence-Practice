import numpy as np

# Problem Definition
def f(x):
    func=(x[2] + 2) * x[1] * (x[0] ** 2)
    g1 = 1 - (((x[1] ** 3) * x[2]) / (7.178 * (x[0] ** 4))) <=0
    g2 = ((4 * (x[1] ** 2) - x[0] * x[1]) / (12.566 * (x[1] * (x[0] ** 3)) - x[0] ** 4)) + (1 / (5.108 * (x[0] ** 2))) - 1 <= 0
    g3 = 1 - ((140.45 * x[0]) / ((x[1] ** 2) * x[2])) <= 0
    g4 = ((x[0] + x[1]) / 1.5) - 1<=0
    g5 = x[0] >= 0.005
    g6 = x[0] <= 2.0
    g7 = x[1] >= 0.25
    g8 = x[1] <= 1.3
    g9 = x[2] >= 2.0
    g10=x[2] <= 15.0
    if g1 and g2 and g3 and g4 and g5 and g6 and g7 and g8 and g9 and g10: 
        return func  # Penalty for violating constraints
    else:
        return 10000


nVar = 3
VarMin = np.array([0.005, 0.25, 2])
VarMax = np.array([2.0, 1.3, 15])

# Firefly Algorithm Parameters
MaxIt = 10
nPop = 350
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
    #print(pop)
    print(pop)
    BestCost[it] = BestSol["Cost"]
    #print(pop)
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
