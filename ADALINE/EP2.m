clear, clc, close all

% ELTON SOARES SILVA - 41711ETE010
% SAMUEL D. LIMA     - 41621ETE011


%% Dados treinamento

% N� de Linhas e Colunas do set de dados de treino
% Nome do arquivo de dados de treino
% Import dos dados de treino
L = 35; C = 5;
filenameTrain = 'dadosTreinamento.csv';
[X, d] = dataTrainIMP(filenameTrain, L, C);


%% Novos dados a serem classificados

% N� de Linhas e Colunas do set de dados NOVOS
% Nome do arquivo de dados NOVOS
% Import dos dados NOVOS
L_NEW = 15; C_NEW = 4;
filenameNew = 'dadosNovos.csv';
X_NEW = dataNewIMP(filenameNew, L_NEW, C_NEW);


%% Treinamento da Rede Neural

% Inicializa a matriz de pesos sin�pticos
W = zeros( 1, C_NEW );

% inicializa��o da taxa de aprendizagem e sigma (patamar aceit�vel)
eta = 0.0025;
sigma = 1E-6;

% uso do algoritmo de treinamento
[w, w_iniciais, epocas, erro] = trainADALINE(X, d, eta, sigma);

% armazena os pesos sin�pticos obtidos em w, em W.
for j = 2:numel(w), W(1, j-1) = w(j); end

% armazenando o limiar obtido em bias
bias = w(1);

%% Aplicando o ADALINE ao conjunto de dados novos

% Inicializa a matriz de sa�da, com as classifica��es de cada amostra
y = zeros( 1, L_NEW );

% obtendo o potencial de ativa��o de cada amostra nova
u = w' * X_NEW;        

% obtendo a classifica��o de cada amostra usando a fun��o degrau
for k = 1:numel(u)
    y(1,k) = degrau(u(1,k));
end

% Mostra na command window as classifica��es obtidas por amostra
y

%% PLOTAGEM DO ERRO M�DIO QUADR�TICO

xErro = 1 : epocas;
plot(xErro, erro, 'r-', 'LineWidth', 2);
set(gca, 'xtick', 1 : floor( epocas / 10 ) : epocas, 'ytick', 0: max(erro) / 10 : max(erro), 'FontSize', 12, 'lineWidth',1.7);
title('Erro Quadr�tico M�dio x �poca de Treinamento')
xlabel('�poca de Treinamento'), ylabel('Erro Quadr�tico M�dio'), grid

%% FUN��O DEGRAU
% Retorna   1, se u >= 0
% Retorna  -1, se u < 0
function y = degrau(u)
        if ( u >= 0 ), y = 1; else, y = -1; end
end