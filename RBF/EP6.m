%% INICIALIZA��O
clear all
close all
clc

% ELTON SOARES SILVA - 41711ETE010
% SAMUEL DAVI DE LIMA - 41621ETE011

%% IMPORT - CONJUNTO DE TREINO E CONJUNTO DE OPERA��O
L = 40; C = 3;
numSaidas = 1;
filenameTrain = 'dadosTreinamento.csv';

% conjunto de opera��o
load('x_new.mat');
load('d_new.mat');

[X, d] = dataTrainIMP(filenameTrain, L, C, numSaidas);

%% TREINO DA REDE
% Inicializa��es
eta = 0.01;
sigma = 1E-7;
maxEpcs = 3000;
model = [2 1]; % 2 camadas, a intermedi�ria tem 2 neur�nios e a de sa�da tem 1

% retorna os pesos finais, iniciais, n� de �pocas gastas, errro m�dio
% quadr�tico por �poca, e a vari�ncia de cada neur�nio.
% recebe os dados de treino, a tx de aprendizado eta, a precis�o sigma, o
% n� m�ximo de �pocas de treino maxEpcs e o modelo da rede (2 neur�nio na
% camada oculta e 1 na camada de sa�da)
[w, w0, epcs, EMQ vari] = trainRBF(X, d, eta, sigma, maxEpcs, model);

%% OPERA��O DA REDE

% usa a rede treinada para encontrar a classifica��o das amostras novas
[resultado] = clas_RBF(w, sqrt(vari), x_new)


