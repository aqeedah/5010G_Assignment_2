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
NUM_RUNS = 10
SEED = 42

# --- Helper Function: Initialize Population ---
def initialize_population(pop_size, dim):
    return np.random.uniform(DOM_MIN, DOM_MAX, (pop_size, dim))

# --- Differential Evolution Function ---
def de_optimize(run_id):
    np.random.seed(SEED)

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

    df = pd.DataFrame({
        "Generation": range(1, MAX_GEN + 1),
        f"Run_{run_id}": best_fitness_list
    })
    return df

# --- Run the Optimization Multiple Times ---
if __name__ == "__main__":
    all_runs = []

    for run in range(NUM_RUNS):
        print(f"Running DE - Run {run + 1}")
        df = de_optimize(run + 1)
        all_runs.append(df.set_index("Generation"))

    # Combine and save
    results_df = pd.concat(all_runs, axis=1)
    results_df.to_csv("DE_multi_run.csv")

    # Plot average convergence
    avg_fitness = results_df.mean(axis=1)
    plt.figure()
    plt.plot(avg_fitness.index, avg_fitness.values, label="DE (avg of runs)", color="orange")
    plt.xlabel("Generation")
    plt.ylabel("Average Best Fitness")
    plt.title(f"Differential Evolution Convergence over {NUM_RUNS} Runs")
    plt.grid()
    plt.savefig("DE_multi_run_plot.png")
    plt.show()
