%% INICIALIZAÇÃO
clear all
close all
clc

% ELTON SOARES SILVA - 41711ETE010
% SAMUEL DAVI DE LIMA - 41621ETE011

%% IMPORT - CONJUNTO DE TREINO E CONJUNTO DE OPERAÇÃO
L = 40; C = 3;
numSaidas = 1;
filenameTrain = 'dadosTreinamento.csv';

% conjunto de operação
load('x_new.mat');
load('d_new.mat');

[X, d] = dataTrainIMP(filenameTrain, L, C, numSaidas);

%% TREINO DA REDE
% Inicializações
eta = 0.01;
sigma = 1E-7;
maxEpcs = 3000;
model = [2 1]; % 2 camadas, a intermediária tem 2 neurônios e a de saída tem 1

% retorna os pesos finais, iniciais, n° de épocas gastas, errro médio
% quadrático por época, e a variância de cada neurônio.
% recebe os dados de treino, a tx de aprendizado eta, a precisão sigma, o
% n° máximo de épocas de treino maxEpcs e o modelo da rede (2 neurônio na
% camada oculta e 1 na camada de saída)
[w, w0, epcs, EMQ vari] = trainRBF(X, d, eta, sigma, maxEpcs, model);

%% OPERAÇÃO DA REDE

% usa a rede treinada para encontrar a classificação das amostras novas
[resultado] = clas_RBF(w, sqrt(vari), x_new)


