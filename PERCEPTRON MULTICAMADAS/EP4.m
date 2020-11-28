%% - Pr�tica 04 PMC
% Nomes: Samuel Davi de Lima e Elton Soares Silva
% N� Mat: 41621ETE011 | 41711ETE010

% OBSERVA��ES
% Entradas -> 04 vari�veis reais, definidas por x1 (teor de �gua), 
% x2 (grau de acidez), x3 (temperatura) e x4 (tens�o superficial). 

%% Inicializa��o
clear all
close all
clc

%% TREINAMENTO

% In�cio
% Import das Amostras de Treinamento
% obs.: J� est�o concatenadas com uma matriz coluna contendo -1,
% representando o multiplicador do bias

nomeArq1 = 'dadosTreinamento.csv';
numRols1 = 130; numCol1 = 7; numSaidas = 3;
[X, d] = dataTrainIMP(nomeArq1, numRols1, numCol1, numSaidas);

% taxa de aprendizagem e precis�o
n = 0.1;
e = 1e-6;

tic
[W1 W2 epocas_gastas Err EM_Atual] = train_PMC(X, d, n, e);
tempoGasto = toc
%% Aplica��o da Rede aos Dados Novos

% Import dos dados novos p/ classifica��o
nomeArq2 = 'dadosNovos.csv';
numRols2 = 18; numCol2 = 4;
xnovos = dataNewIMP(nomeArq2, numRols2, numCol2);

% Aplica o PMC aos dados novos, com sa�da sendo y, os dados com p�s
% processamento e yNoPos sendo a sa�da sem p�s-processamento
[y, yNoPos] = clas_PMC(xnovos, W1, W2);

% arredonda a sa�da sem p�s-processamento e converte para texto
% para ficar f�cil de copiar para a tabela do relat�rio no word
yNoPos = round(yNoPos, 3);
yNoPos2 = arrayfun(@num2str, yNoPos, 'un', 0);