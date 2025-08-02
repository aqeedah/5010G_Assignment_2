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

# --- Particle Swarm Optimization Function ---
def pso_optimize(run_id):
    np.random.seed(SEED)

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

    df = pd.DataFrame({
        "Generation": range(1, MAX_GEN + 1),
        f"Run_{run_id}": best_fitness_list
    })
    return df

# --- Run the Optimization Multiple Times ---
if __name__ == "__main__":
    all_runs = []

    for run in range(NUM_RUNS):
        print(f"Running PSO - Run {run + 1}")
        df = pso_optimize(run + 1)
        all_runs.append(df.set_index("Generation"))

    # Combine and save
    results_df = pd.concat(all_runs, axis=1)
    results_df.to_csv("PSO_multi_run.csv")

    # Plot average convergence
    avg_fitness = results_df.mean(axis=1)
    plt.figure()
    plt.plot(avg_fitness.index, avg_fitness.values, label="PSO (avg of runs)", color="green")
    plt.xlabel("Generation")
    plt.ylabel("Average Best Fitness")
    plt.title(f"PSO Convergence over {NUM_RUNS} Runs")
    plt.grid()
    plt.savefig("PSO_multi_run_plot.png")
    plt.show()
