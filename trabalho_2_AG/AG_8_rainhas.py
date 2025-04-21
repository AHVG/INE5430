import random
import matplotlib.pyplot as plt

TAMANHO_POP = 10000
GERACOES = 500
TAXA_MUTACAO = 0.1
NUM_GENES = 8
META_FITNESS = 28


def fitness(individuo):
    conflitos = 0
    for i in range(len(individuo)):
        for j in range(i + 1, len(individuo)):
            if individuo[i] == individuo[j] or abs(individuo[i] - individuo[j]) == abs(i - j):
                conflitos += 1
    return META_FITNESS - conflitos

def gerar_individuo():
    return [random.randint(0, 7) for _ in range(NUM_GENES)]

def crossover(pai1, pai2):
    ponto = random.randint(1, NUM_GENES - 2)
    return pai1[:ponto] + pai2[ponto:]

def mutacao(individuo):
    novo = individuo[:]
    if random.random() < TAXA_MUTACAO:
        col = random.randint(0, NUM_GENES - 1)
        novo[col] = random.randint(0, 7)
    return novo

populacao = [gerar_individuo() for _ in range(TAMANHO_POP)]
melhor_fitness_geracao = []

for geracao in range(GERACOES):
    fitnesses = [fitness(ind) for ind in populacao]

    if max(fitnesses) == META_FITNESS:
        melhor_fitness_geracao.append(max(fitnesses))
        break

    # Seleciona a elite (top 50%)
    combinados = list(zip(populacao, fitnesses))
    combinados.sort(key=lambda x: x[1], reverse=True)
    elite = [ind for ind, _ in combinados[:TAMANHO_POP // 2]]

    # Gera filhos a partir apenas da elite
    filhos = []
    while len(filhos) < TAMANHO_POP // 2:
        pai1 = random.choice(elite)
        pai2 = random.choice(elite)
        filho = crossover(pai1, pai2)
        filho = mutacao(filho)
        filhos.append(filho)

    populacao = elite + filhos
    melhor_fitness_geracao.append(max(fitnesses))

# Plot da convergência
plt.plot(melhor_fitness_geracao)
plt.title("Convergência do Algoritmo Genético")
plt.xlabel("Geração")
plt.ylabel("Melhor Fitness")
plt.grid(True)
plt.savefig("convergencia.png")

# Resultado final
melhor_indice = fitnesses.index(max(fitnesses))
print("Melhor solução encontrada:", populacao[melhor_indice])
print("Fitness:", fitnesses[melhor_indice])
print("Geração:", len(melhor_fitness_geracao))
