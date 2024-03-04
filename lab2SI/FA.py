import numpy as np
from numpy.random import default_rng
ranges=[(2.6, 3.6), (0.7, 0.8), (17, 28), (7.3, 8.3), (7.8, 8.3), (2.9, 3.9), (5.0, 5.5)]
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
class FireflyAlgorithm:
    def __init__(self, pop_size=20, alpha=1.0, betamin=1.0, gamma=0.01, seed=None):
        self.pop_size = pop_size
        self.alpha = alpha
        self.betamin = betamin
        self.gamma = gamma
        self.rng = default_rng(seed)

    def run(self, function, dim, ranges, max_evals):
        fireflies = self.rng.uniform([r[0] for r in ranges], [r[1] for r in ranges], size=(250, 7))
        intensity = np.apply_along_axis(function, 1, fireflies)
        best = np.min(intensity)

        evaluations = self.pop_size
        new_alpha = self.alpha
        search_range = ub - lb

        while evaluations <= max_evals:
            new_alpha *= 0.97
            for i in range(self.pop_size):
                for j in range(self.pop_size):
                    if intensity[i] >= intensity[j]:
                        r = np.sum(np.square(fireflies[i] - fireflies[j]), axis=-1)
                        beta = self.betamin * np.exp(-self.gamma * r)
                        steps = new_alpha * (self.rng.random(dim) - 0.5) * search_range
                        fireflies[i] += beta * (fireflies[j] - fireflies[i]) + steps
                        fireflies[i] = np.clip(fireflies[i], lb, ub)
                        intensity[i] = function(fireflies[i])
                        evaluations += 1
                        best = min(intensity[i], best)
        return best