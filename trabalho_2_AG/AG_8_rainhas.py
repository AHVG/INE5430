import pygad


# Problema das 8 rainhas
N = 8
# Cálculo do número máximo de pares de rainhas que não se atacam
MAX_NON_ATTACKING_PAIRS = (N * (N - 1)) // 2

# Função de fitness: conta quantos pares de rainhas NÃO se atacam
def fitness_func(ga_instance, solution, solution_idx):
    non_attacking = MAX_NON_ATTACKING_PAIRS
    for i in range(N):
        for j in range(i + 1, N):
            same_row = solution[i] == solution[j]
            same_diag = abs(solution[i] - solution[j]) == abs(i - j)
            if same_row or same_diag:
                non_attacking -= 1
    return non_attacking

# Espaço dos genes: cada gene representa uma linha da rainha em cada coluna
gene_space = list(range(N))

# Criação da instância do algoritmo genético
ga_instance = pygad.GA(
    num_generations=2000,
    num_parents_mating=N * 2,
    fitness_func=fitness_func,
    sol_per_pop=100,
    num_genes=N,
    gene_space=gene_space,
    parent_selection_type="tournament",
    crossover_type="single_point",
    mutation_type="random",
    mutation_percent_genes=20,
    stop_criteria=[f"reach_{MAX_NON_ATTACKING_PAIRS}"]
)

# Executa o algoritmo genético
ga_instance.run()

# Plota a evolução da fitness
ga_instance.plot_fitness()

# Mostra a melhor solução encontrada
solution, solution_fitness, _ = ga_instance.best_solution()
print(f"\nMelhor solução encontrada (N={N}):", solution)
print("Fitness da melhor solução:", solution_fitness)

# Imprime o tabuleiro com a solução
def print_board(solution):
    board = [['.' for _ in range(N)] for _ in range(N)]
    for col, row in enumerate(solution):
        board[int(row)][col] = 'Q'
    print("\nTabuleiro:")
    for row in board:
        print(' '.join(row))

print_board(solution)
