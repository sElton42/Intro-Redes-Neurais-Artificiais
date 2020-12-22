% ELTON SOARES SILVA - 41711ETE010

clear, clc, close all

%% Import dos Dados

% N° de Linhas e Colunas do set de dados
L = 768; C = 9;
numSaidas = 1;
% Nome do arquivo de dados
filenameTrain = 'diabetes.csv';
% Import dos dados
[X, d] = dataImport(filenameTrain, L, C, numSaidas);

% Faz a transposta por questões de debug
X = X';

% Elimina amostras onde uma ou mais entradas estão faltando valores
counter = 0;
for nSample = 1 : L
    if( (X(nSample, 3) ~= 0) & (X(nSample, 4) ~= 0) & (X(nSample, 5) ~= 0) & (X(nSample, 6) ~= 0) & (X(nSample, 7) ~= 0) )
        counter = counter + 1;
        X_(counter, :) = X(nSample, :);
        d_(1, counter) = d(1, nSample);
    end
end

% desfaz a transposta
X_ = X_';

% normalização
z = X_;
for nSample = 1 : size(X_,2)
    for nLine = 2 : size(X_,1)
        z(nLine,nSample) = ( X_(nLine, nSample) - min( X_(nLine, :) ) ) / ( max( X_(nLine, :) ) - min( X_(nLine, :) ) );
    end
end

% Separa o set de treino do set de teste
zTreino     = z(:, 1:275);
dTreino     = d_(1, 1:275);
zTeste      = z(:, 276:392);
dTeste      = d_(1, 276:392);

%% Pré-processamento das saídas para uso com RBF usando -1 e 1
for j = 1:numel(dTreino)
    if(dTreino(1, j) == 0)
        dTreino(1, j) = -1;
    end
end

for j = 1:numel(dTeste)
    if(dTeste(1, j) == 0)
        dTeste(1, j) = -1;
    end
end

%% Pré-Processamento das saídas para usar dois neurônios na camada de saída

dTreino(2,:) = zeros( 1,numel(dTreino) );
for j = 1 : size(dTreino, 2)
    if( dTreino(1,j) == -1 )
        dTreino(1,j) = 0;
        dTreino(2,j) = 1;
    end
end

dTeste(2,:) = zeros( 1,numel(dTeste) );
for j = 1 : size(dTeste, 2)
    if( dTeste(1,j) == -1 )
        dTeste(1,j) = 0;
        dTeste(2,j) = 1;
    end
end

%% TREINO DA REDE

% Inicializações
eta = 0.001;
sigma = 1E-6;
maxEpcs = 30000;
model = [21 2];

tic
[w, w0, epcs, EMQ, accuracyTrain, vari] = trainRBF(zTreino, dTreino, eta, sigma, maxEpcs, model);
tempoGasto = toc

%% OPERAÇÃO DA REDE NO CONJUNTO DE TESTES

%       APLICAÇÃO DAS ENTRADAS NA CAMADA OCULTA
for nPeso = 1:model(1)
    for nAmost = 1 : size(zTeste,2)
        %               aplica a função de ativação para cada amostra em cada
        %               neurônio.
        gu{nPeso,1}(1, nAmost) = exp( (-1) * norm( zTeste(2:size(zTeste,1), nAmost) - w{1,1}(nPeso, :)' )^2 / ( 2*vari(nPeso, 1) ) );
    end
end

%       APLICAÇÃO DAS SAÍDAS DA CAMADA OCULTA NA CAMADA DE SAÍDA
for nAmost2 = 1:size(zTeste,2)
    for nPeso2 = 1:model(1)
        vetorAmostras(nPeso2, nAmost2) = gu{nPeso2, 1}(1,nAmost2);
    end
end

vetorAmostras_bias = [ (-1)*ones( 1, size(vetorAmostras, 2) ) ; vetorAmostras];

saidaTeste = w{2,1} * vetorAmostras_bias;

%% CHECAGEM ACERTOS (1n)
% acertosTeste = 0;
% for nAmos = 1:size(dTeste, 2)
%     if( (saidaTeste(1,nAmos) >= 0 & dTeste(1,nAmos) >= 0) | (saidaTeste(1,nAmos) < 0 & dTeste(1,nAmos) < 0) ), acertosTeste = acertosTeste + 1; end
% end
% acertosTeste
% acertosTeste / size(dTeste, 2) * 100
% epcs

%% CHECAGEM ACERTOS (2n)
acertosTeste = 0;
for nAmos = 1:size(dTeste, 2)
    if ( (ceil(saidaTeste(1, nAmos)) == dTeste(1, nAmos) & floor(saidaTeste(2, nAmos)) == dTeste(2, nAmos)) | (floor(saidaTeste(1, nAmos)) == dTeste(1, nAmos) & ceil(saidaTeste(2, nAmos)) == dTeste(2, nAmos)) )
        acertosTeste = acertosTeste + 1;
    end
end
acertosTeste
acertosTeste / size(dTeste, 2) * 100
epcs
%% PLOTAGEM DO ERRO MÉDIO QUADRÁTICO

plot(EMQ, 'r-', 'LineWidth', 2);
set(gca, 'xtick', 1 : floor( epcs / 10 ) : epcs, 'ytick', 0: max(EMQ) / 10 : max(EMQ), 'FontSize', 12, 'lineWidth',1.7);
title('Erro Quadrático Médio x Época de Treinamento')
xlabel('Época de Treinamento'), ylabel('Erro Quadrático Médio'), grid
