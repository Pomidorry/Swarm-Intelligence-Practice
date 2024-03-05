import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#from matplotlib.animation import FuncAnimation

# Problem Definition
def rosenbrock2(x):
    condition = np.array(x[0])**2 + np.array(x[1])**2 >= 2
    
    func_values = np.array((1-x[0])**2 + 100*(x[1]-x[0]**2)**2)
    func_values[condition] = 10000
    return func_values


nVar = 2
VarMin = np.array([-1.25, -1.25])
VarMax = np.array([1.25, 1.25])

# Firefly Algorithm Parameters
MaxIt = 50
nPop = 20
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
    pop[i]["Cost"] = rosenbrock2(pop[i]["Position"])
    if pop[i]["Cost"] <= BestSol["Cost"]:
        BestSol = pop[i].copy()

BestCost = np.zeros(MaxIt)
pops=[]
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

                newsol["Cost"] = rosenbrock2(newsol["Position"])

                if newsol["Cost"] <= newpop[i]["Cost"]:
                    newpop[i] = newsol.copy()
                    if newpop[i]["Cost"] <= BestSol["Cost"]:
                        BestSol = newpop[i].copy()

    pop = np.concatenate((pop, newpop))

    SortOrder = np.argsort([pop[i]["Cost"] for i in range(len(pop))])
    pop = pop[SortOrder]

    pop = pop[:nPop]
    pops.append(pop)
    BestCost[it] = BestSol["Cost"]

    print(f'Iteration {it}: Best Cost = {BestCost[it]}')

    alpha *= alpha_damp

# Results
# Results
plt.figure()
plt.semilogy(BestCost, linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Best Cost')
plt.grid(True)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X = np.arange(VarMin[0], VarMax[0], 0.01)
Y = np.arange(VarMin[1], VarMax[1], 0.01)
X, Y = np.meshgrid(X, Y)
Z = rosenbrock2([X, Y])
ax.plot_wireframe(X, Y, Z, rstride=30, cstride=30, color='c', alpha=0.6)
        
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Fitness')
ax.set_title('FA')
# # Animation
# fig, ax = plt.subplots()
# xdata, ydata = [], []
# ln, = plt.plot([], [], 'ro', animated=True)

# def init():
#     ax.set_xlim(0, MaxIt)
#     ax.set_ylim(0, max(BestCost))
#     return ln,
#print(pops[0][0]["Position"])
def update(frame):
    ax = fig.add_subplot(111, projection='3d')
    X = np.arange(-5, 5, 0.01)
    Y = np.arange(-5, 5, 0.01)
    X, Y = np.meshgrid(X, Y)
    Z = rosenbrock2([X, Y])
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.2)
    for n in range(nPop):
        ax.scatter(pops[frame][n]["Position"], pops[frame][n]["Cost"], marker='*', edgecolors='red')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Fitness')
    ax.set_title('FA')

ani = animation.FuncAnimation(fig, update, frames=range(MaxIt), interval=200)

plt.show()