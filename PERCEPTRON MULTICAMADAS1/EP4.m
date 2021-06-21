%% - Prática 04 PMC
% Nomes: Samuel Davi de Lima e Elton Soares Silva
% N° Mat: 41621ETE011 | 41711ETE010

% OBSERVAÇÕES
% Entradas -> 04 variáveis reais, definidas por x1 (teor de água), 
% x2 (grau de acidez), x3 (temperatura) e x4 (tensão superficial). 

%% Inicialização
clear all
close all
clc

%% TREINAMENTO

% Início
% Import das Amostras de Treinamento
% obs.: Já estão concatenadas com uma matriz coluna contendo -1,
% representando o multiplicador do bias

nomeArq1 = 'dadosTreinamento.csv';
numRols1 = 130; numCol1 = 7; numSaidas = 3;
[X, d] = dataTrainIMP(nomeArq1, numRols1, numCol1, numSaidas);

% taxa de aprendizagem e precisão
n = 0.1;
e = 1e-6;

tic
[W1 W2 epocas_gastas Err EM_Atual] = train_PMC(X, d, n, e);
tempoGasto = toc
%% Aplicação da Rede aos Dados Novos

% Import dos dados novos p/ classificação
nomeArq2 = 'dadosNovos.csv';
numRols2 = 18; numCol2 = 4;
xnovos = dataNewIMP(nomeArq2, numRols2, numCol2);

% Aplica o PMC aos dados novos, com saída sendo y, os dados com pós
% processamento e yNoPos sendo a saída sem pós-processamento
[y, yNoPos] = clas_PMC(xnovos, W1, W2);

% arredonda a saída sem pós-processamento e converte para texto
% para ficar fácil de copiar para a tabela do relatório no word
yNoPos = round(yNoPos, 3);
yNoPos2 = arrayfun(@num2str, yNoPos, 'un', 0);