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

# --- Differential Evolution Function ---
def de_optimize():
    F = 0.8  # Mutation factor
    CR = 0.9  # Crossover probability
    pop = initialize_population(POP_SIZE, DIM)
    fitness = np.array([rastrigin(ind) for ind in pop])
    best_fitness_list = []

    for gen in range(MAX_GEN):
        for i in range(POP_SIZE):
            idxs = [idx for idx in range(POP_SIZE) if idx != i]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), DOM_MIN, DOM_MAX)
            cross_points = np.random.rand(DIM) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(DIM)] = True
            trial = np.where(cross_points, mutant, pop[i])
            trial = np.clip(trial, DOM_MIN, DOM_MAX)
            f_trial = rastrigin(trial)
            if f_trial < fitness[i]:
                pop[i], fitness[i] = trial, f_trial
        best_fitness_list.append(np.min(fitness))

    pd.DataFrame({"Generation": range(1, MAX_GEN+1), "BestFitness": best_fitness_list}).to_csv("DE.csv", index=False)

    # Plot convergence
    plt.figure()
    plt.plot(range(1, MAX_GEN+1), best_fitness_list, label="DE", color="orange")
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("Differential Evolution Convergence")
    plt.grid()
    plt.savefig("DE_plot.png")
    plt.show()

    best_idx = np.argmin(fitness)
    return pop[best_idx], fitness[best_idx]

# --- Run the Optimization ---
if __name__ == "__main__":
    best_sol, best_fit = de_optimize()
    print(f"Best Solution: {best_sol}, Best Fitness: {best_fit}")
