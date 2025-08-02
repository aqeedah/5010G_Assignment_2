import numpy as np
import random
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

# --- Genetic Algorithm Function ---
def ga_optimize(run_id):
    random.seed(SEED)
    np.random.seed(SEED)

    mutation_rate = 0.1
    crossover_rate = 0.7

    def tournament_selection(pop, fitness, k=3):
        idx = np.random.choice(len(pop), k)
        winner = idx[np.argmin(fitness[idx])]
        return pop[winner]

    pop = initialize_population(POP_SIZE, DIM)
    fitness = np.array([rastrigin(ind) for ind in pop])
    best_fitness_list = []

    for gen in range(MAX_GEN):
        new_pop = []
        for _ in range(POP_SIZE):
            parent1 = tournament_selection(pop, fitness)
            parent2 = tournament_selection(pop, fitness)

            # Crossover
            if np.random.rand() < crossover_rate:
                alpha = np.random.rand()
                child = alpha * parent1 + (1 - alpha) * parent2
            else:
                child = parent1.copy()

            # Mutation
            if np.random.rand() < mutation_rate:
                mut_dim = np.random.randint(DIM)
                child[mut_dim] = np.random.uniform(DOM_MIN, DOM_MAX)

            new_pop.append(np.clip(child, DOM_MIN, DOM_MAX))

        pop = np.array(new_pop)
        fitness = np.array([rastrigin(ind) for ind in pop])
        best_fitness_list.append(np.min(fitness))

    # Save results per run
    df = pd.DataFrame({
        "Generation": range(1, MAX_GEN+1),
        f"Run_{run_id}": best_fitness_list
    })
    return df

# --- Run the Optimization Multiple Times ---
if __name__ == "__main__":
    all_runs = []

    for run in range(NUM_RUNS):
        print(f"Running GA - Run {run + 1}")
        df = ga_optimize(run + 1)
        all_runs.append(df.set_index("Generation"))

    # Combine and save
    results_df = pd.concat(all_runs, axis=1)
    results_df.to_csv("GA_multi_run.csv")

    # Plot average convergence
    avg_fitness = results_df.mean(axis=1)
    plt.figure()
    plt.plot(avg_fitness.index, avg_fitness.values, label="GA (avg of runs)", color="blue")
    plt.xlabel("Generation")
    plt.ylabel("Average Best Fitness")
    plt.title(f"GA Convergence over {NUM_RUNS} Runs")
    plt.grid()
    plt.savefig("GA_multi_run_plot.png")
    plt.show()
