from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# 1. Definindo a estrutura da rede
model = BayesianModel([
    ('E', 'C'),
    ('E', 'V'),
    ('E', 'S'),
    ('C', 'P'),
    ('S', 'P')
])

# 2. CPDs

cpd_E = TabularCPD('E', 3, [[0.6], [0.3], [0.1]], state_names={'E': ['Fundamental', 'Secundário', 'Universitário']})

cpd_C = TabularCPD(
    'C', 2,
    [[1.0, 0.2, 0.4],  # C = Não
     [0.0, 0.8, 0.6]],  # C = Sim
    evidence=['E'],
    evidence_card=[3],
    state_names={'C': ['Não', 'Sim'], 'E': ['Fundamental', 'Secundário', 'Universitário']}
)

cpd_V = TabularCPD(
    'V', 2,
    [[0.9, 0.0, 0.2],  # V = Não
     [0.1, 1.0, 0.8]],  # V = Sim
    evidence=['E'],
    evidence_card=[3],
    state_names={'V': ['Não', 'Sim'], 'E': ['Fundamental', 'Secundário', 'Universitário']}
)

cpd_S = TabularCPD(
    'S', 2,
    [[1.0, 0.5, 0.5],  # S = Não
     [0.0, 0.5, 0.5]],  # S = Sim
    evidence=['E'],
    evidence_card=[3],
    state_names={'S': ['Não', 'Sim'], 'E': ['Fundamental', 'Secundário', 'Universitário']}
)

cpd_P = TabularCPD(
    'P', 2,
    [
        [1.0, 0.99, 1.0, 0.90],  # P = Não
        [0.0, 0.01, 0.0, 0.10]   # P = Sim
    ],
    evidence=['C', 'S'],
    evidence_card=[2, 2],
    state_names={'P': ['Não', 'Sim'], 'C': ['Não', 'Sim'], 'S': ['Não', 'Sim']}
)

# 3. Adicionar CPDs ao modelo
model.add_cpds(cpd_E, cpd_C, cpd_V, cpd_S, cpd_P)

# 4. Verificar o modelo
assert model.check_model()

# 5. Inferência
infer = VariableElimination(model)

# a) Qual a probabilidade de um aluno colar?
prob_cola = infer.query(variables=['C'], show_progress=False)
print("Probabilidade de colar:")
print(prob_cola)

# b) Qual a probabilidade de um aluno frequentar o Ensino Secundário dado que viu colegas colando e se sentiu penalizado?
prob_secundario = infer.query(
    variables=['E'],
    evidence={'V': 'Sim', 'P': 'Sim'},
    show_progress=False
)
print("\nProbabilidade de ser aluno do Ensino Secundário dado que viu colegas colando e se sentiu penalizado:")
print(prob_secundario)
