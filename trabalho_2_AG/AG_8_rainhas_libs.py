import pygad


def fitness_func(ga_instance, solution, solution_idx):
    non_attacking = 28
    for i in range(len(solution)):
        for j in range(i + 1, len(solution)):
            if solution[i] == solution[j] or abs(solution[i] - solution[j]) == abs(i - j):
                non_attacking -= 1
    return non_attacking

gene_space = list(range(8))

ga_instance = pygad.GA(
    num_generations=1000,
    num_parents_mating=20,
    fitness_func=fitness_func,
    sol_per_pop=100,
    num_genes=8,
    gene_space=gene_space,
    parent_selection_type="tournament",
    crossover_type="single_point",
    mutation_type="random",
    mutation_percent_genes=20,
    stop_criteria=["reach_28"]
)

ga_instance.run()
ga_instance.plot_fitness()

solution, solution_fitness, _ = ga_instance.best_solution()
print(f"Melhor solução encontrada: {solution}")
print(f"Fitness da melhor solução: {solution_fitness}")

# _ _ _ 3 _ _ _ _
# _ _ _ _ _ 5 _ _
# _ _ _ _ _ _ _ 7
# _ 1 _ _ _ _ _ _
# _ _ _ _ _ _ 6 _
# 0 _ _ _ _ _ _ _
# _ _ 2 _ _ _ _ _
# _ _ _ _ 4 _ _ _
