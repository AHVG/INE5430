% Base de conhecimento: sintomas associados às doenças
doenca(infarto_do_miocardio, [dor_no_peito, dor_no_braco_esquerdo, falta_de_ar, suor_frio, nauseas, desmaios]).
doenca(insuficiencia_cardiaca, [falta_de_ar_ao_deitar, fadiga, inchaco, ganho_de_peso, tosse_noturna]).
doenca(arritmia_cardiaca, [palpitacoes, tontura, falta_de_ar, ansiedade_subita]).
doenca(doenca_arterial_periferica, [claudicacao, feridas_nos_pes, pele_fria]).
doenca(valvopatia, [fadiga, tontura, desmaios, barulho_no_coracao, dor_no_peito, inchaco]).
doenca(hipertensao_arterial, [sangramento_nasal, visao_embacada, dor_de_cabeca, zumbido, tontura]).
doenca(aterosclerose, [formigamento, claudicacao, dor_no_peito]).
doenca(aneurisma, [pulsacao_abdominal, nauseas, choque, dor_subita]).

% Base de conhecimento: tratamento para cada doença
tratamento(infarto_do_miocardio, [aspirina, nitroglicerina, tromboliticos, angioplastia]).
tratamento(insuficiencia_cardiaca, [diureticos, ieca, betabloqueadores]).
tratamento(arritmia_cardiaca, [anticoagulantes, antiarritmicos, cardioversao, ablacacao]).
tratamento(doenca_arterial_periferica, [antiplaquetarios, estatinas, cirurgia, mudanca_de_estilo_de_vida]).
tratamento(valvopatia, [substituicao_valvar, betabloqueadores, diureticos, anticoagulantes]).
tratamento(hipertensao_arterial, [diureticos, ieca, mudanca_de_estilo_de_vida]).
tratamento(aterosclerose, [angioplastia, antiplaquetarios, estatinas]).
tratamento(aneurisma, [cirurgia, estatinas, betabloqueadores]).

% Regras: identificação de doença por sintomas
diagnostico(SintomasPaciente, Doenca) :-
    doenca(Doenca, SintomasDoenca),
    subset(SintomasPaciente, SintomasDoenca).

% Sugere o tratamento para a doença identificada
sugerir_tratamento(Sintomas, Tratamento) :-
    diagnostico(Sintomas, Doenca),
    tratamento(Doenca, Tratamento).

% ------------------------
% Exemplos de uso:
% ------------------------

% ?- diagnostico([dor_no_peito, suor_frio, desmaios, falta_de_ar, dor_no_braco_esquerdo, nauseas], D).
% D = infarto_do_miocardio.

% ?- sugerir_tratamento([dor_no_peito, suor_frio, desmaios, falta_de_ar, dor_no_braco_esquerdo, nauseas], T).
% T = [aspirina, nitroglicerina, tromboliticos, angioplastia].

% ?- diagnostico([palpitacoes, tontura, ansiedade_subita, falta_de_ar], D).
% D = arritmia_cardiaca.

% ?- diagnostico([claudicacao, dor_no_peito, formigamento], D).
% D = aterosclerose.

% ?- sugerir_tratamento([claudicacao, dor_no_peito, formigamento], T).
% T = [colesterol, angioplastia, antiplaquetarios, estatinas].

