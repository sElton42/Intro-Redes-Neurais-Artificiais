clear, clc, close all

% ELTON SOARES SILVA - 41711ETE010
% SAMUEL D. LIMA     - 41621ETE011

% AMOSTRAS NOVAS CLASSIFICADAS COMO -1: 1 7 9 10
% AMOSTRAS NOVAS CLASSIFICADAS COMO  1: 2 3 4 5 6 8

%% Dados treinamento

% Entradas
X = [-0.6508,-1.4492,2.0850,0.2626,0.6418,0.2569,1.1155,0.0914,0.0121,-0.0429,0.4340,0.2735,0.4839,0.4089,1.4391,...
    -0.9115,0.3654,0.2144,0.2013,0.6483,-0.1147,-0.7970,-1.0625,0.5307,-1.2200,0.3957,-0.1013,2.4482,...
    2.0149,0.2012;0.1097,0.8896,0.6876,1.1476,1.0234,0.6730,0.6043,0.3399,0.5256,0.4660,0.6870,1.0287,0.4851,...
    -0.1267,0.1614,-0.1973,1.0475,0.7515,1.0014,0.2183,0.2242,0.8795,0.6366,0.1285,0.7777,0.1076,0.5989,0.9455,...
    0.6192,0.2611;4.0009,4.4005,12.0710,7.7985,7.0427,8.3265,7.4446,7.0677,4.6316,5.4323,8.2287,7.1934,7.4850,...
    5.5019,8.5843,2.1962,7.4858,7.1699,6.5489,5.8991,7.2435,3.8762,2.4707,5.6883,1.7252,5.6623,7.1812,11.2095,...
    10.9263,5.4631];

% Saídas
d = [-1.0000,-1.0000,-1.0000,1.0000,1.0000,-1.0000,1.0000,-1.0000,1.0000,1.0000,-1.0000,1.0000,-1.0000,-1.0000...
    ,-1.0000,-1.0000,1.0000,1.0000,1.0000,1.0000,-1.0000,1.0000,1.0000,1.0000,1.0000,-1.0000,-1.0000,1.0000,...
    -1.0000,1.0000];

% Concatena matriz linha de uns com a matriz de dados
[linha, coluna] = size(X);
uns = ones(1, coluna);
X = [uns ; X];

%% Novos dados a serem classificados

% Dados das amostras por entradas (entrada x1, x2 e x3)
x1n = [-0.3565,-0.7842,0.3012,0.7757,0.1570,-0.7014,0.3748,-0.6920,-1.3970,-1.8842];
x2n = [0.0620,1.1267,0.5611,1.0648,0.8028,1.0316,0.1536,0.9404,0.7141,-0.2805];
x3n = [5.9891,5.5912,5.8234,8.0677,6.3040,3.6005,6.1537,4.4058,4.9263,1.2548];

% Concatena matriz linha de uns com as matrizes de dados novos
n2 = numel(x1n);
uns2 = ones(1, n2);
xnews = [uns2; x1n ; x2n ; x3n];


%% Treinamento da Rede Neural

% taxa de aprendizagem de 0,01
eta = 0.01;

% uso do algoritmo de treinamento
[w, w_iniciais, epocas] = trainPercep(X, d, eta);

% armazenando os pesos obtidos em W
W = [w(2) w(3) w(4)];

% armazenando o limiar obtido em bias
bias = w(1);

%% Aplicando o Perceptron ao conjunto de dados novos

% obtendo o potencial de ativação de cada amostra
u = w' * xnews;        

% obtendo a classificação de cada amostra usando a função degrau
for k = 1:numel(u)
    y(1,k) = degrau(u(1,k));
end
% Mostra na command window as classificações obtidas por amostra
y

%% PLOTAGEM DADOS DE TREINO E DADOS NOVOS COM PLANO DE SEPARAÇÃO DE CLASSES (JÁ QUE O PROBLEMA ENVOLVE APENAS 3 ENTRADAS)

% plota dados de treino em azul
plot3(X(2,:), X(3,:), X(4,:), 'bo', 'LineWidth', 4); hold on;

% plota dados novos em verde
plot3(x1n, x2n, x3n, 'go', 'LineWidth', 4)

% plota o plano de separação das classes
plotpc(W, bias)
grid

% OBS.: caso não se queira usar a toolbox de redes neurais do matlab,
% pode-se usar os seguintes comandos para plotar o plano de separação:
% 
% -----------------------------------------------------
% XP = min(X(2,:)) - 10 : 0.2 : max(X(2,:)) + 10;
% YP = min(X(3,:)) - 10 : 0.2 : max(X(3,:)) + 10;
% 
% [XMESH, YMESH] = meshgrid(XP, YP);
% 
% ZP = -(W(1) .* XMESH + W(2) .* YMESH + bias) ./ W(3);
% 
% mesh(XMESH, YMESH, ZP)
% -----------------------------------------------------

%% FUNÇÃO DEGRAU
% Retorna   1, se u >= 0
% Retorna  -1, se u < 0
function y = degrau(u)
        if ( u >= 0 ) y = 1; else y = -1; end
end