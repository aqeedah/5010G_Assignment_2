import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- Rastrigin Function ---
def rastrigin(X):
    x, y = X
    return 20 + x**2 + y**2 - 10 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))

# --- Parameters ---
DOM_MIN, DOM_MAX = -5.12, 5.12
DIM = 2
POP_SIZE = 50
MAX_GEN = 100
SEED = 42
np.random.seed(SEED)

# --- Helper Function: Initialize Population ---
def initialize_population(pop_size, dim):
    return np.random.uniform(DOM_MIN, DOM_MAX, (pop_size, dim))

# --- Particle Swarm Optimization Function ---
def pso_optimize():
    w = 0.7    # Inertia weight
    c1 = 1.4   # Cognitive component
    c2 = 1.4   # Social component

    X = initialize_population(POP_SIZE, DIM)
    V = np.random.uniform(-1, 1, (POP_SIZE, DIM))
    pbest = X.copy()
    pbest_val = np.array([rastrigin(ind) for ind in X])
    gbest_idx = np.argmin(pbest_val)
    gbest = pbest[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]
    best_fitness_list = []

    for gen in range(MAX_GEN):
        r1 = np.random.rand(POP_SIZE, DIM)
        r2 = np.random.rand(POP_SIZE, DIM)
        V = w * V + c1 * r1 * (pbest - X) + c2 * r2 * (gbest - X)
        X = X + V
        X = np.clip(X, DOM_MIN, DOM_MAX)
        fit = np.array([rastrigin(ind) for ind in X])
        better = fit < pbest_val
        pbest[better] = X[better]
        pbest_val[better] = fit[better]
        gbest_idx = np.argmin(pbest_val)
        gbest = pbest[gbest_idx]
        gbest_val = pbest_val[gbest_idx]
        best_fitness_list.append(gbest_val)

    pd.DataFrame({"Generation": range(1, MAX_GEN+1), "BestFitness": best_fitness_list}).to_csv("PSO.csv", index=False)

    # Plot convergence
    plt.figure()
    plt.plot(range(1, MAX_GEN+1), best_fitness_list, label="PSO", color="green")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("Particle Swarm Optimization Convergence")
    plt.grid()
    plt.savefig("PSO_plot.png")
    plt.show()

    return gbest, gbest_val

# --- Run the Optimization ---
if __name__ == "__main__":
    best_sol, best_fit = pso_optimize()
    print(f"Best Solution: {best_sol}, Best Fitness: {best_fit}")
