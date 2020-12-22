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
        z(nLine,nSample) = 2 * ( X_(nLine, nSample) - min( X_(nLine, :) ) ) / ( max( X_(nLine, :) ) - min( X_(nLine, :) ) ) - 1;
    end
end

% Separa o set de treino do set de teste
zTreino     = z(:, 1:275);
dTreino     = d_(1, 1:275);
zTeste      = z(:, 276:392);
dTeste      = d_(1, 276:392);

%% Pré-processamento das saídas para uso com Função TANH
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

%% Treinamento da Rede Neural usando o PMC

% inicialização da taxa de aprendizagem e sigma (patamar aceitável)
eta = 0.1;
sigma = 1E-6;
maxEpcs = 30000;
model = [25 2];

% uso do algoritmo de treinamento PMC com Função TANH
tic
[w, w_iniciais, epocas, EMQ, accuracyTrain] = BackPropOnPMCTANH(zTreino, dTreino, eta, sigma, maxEpcs, model);
tempoGasto = toc

%% Aplicando o PMC com Função TANH

for numAmostra = 1 : size( (zTeste) , 2)
    amostra = zTeste(:,numAmostra);
    amostra = amostra';
    saidaDesejada = dTeste(:, numAmostra);
    
    for numLayer = 1:numel(model)
        w_ = w{numLayer, 1};
        w_ = w_';
        
        if(numLayer == 1)
            u_ = amostra * w_;
            u{1,1}(:,numAmostra) = u_';
            y_ = ( 1 - exp(-u_) ) ./ ( 1 + exp( -u_ ) );
            y{1,1}(:,numAmostra) = y_';
            y_ = [-1 , y_ ];
        else
            u_ = y_ * w_;
            u{numLayer,1}(:,numAmostra) = u_';
            y_ = ( 1 - exp(-u_) ) ./ ( 1 + exp( -u_ ) );
            y{numLayer,1}(:,numAmostra) = y_';
            y_ = [-1 , y_ ];
        end
    end
end

%% CHECAGEM DE ACERTOS DO TREINAMENTO COM F. TANH
acertosTeste = 0;
for nAmos = 1:size(dTeste, 2)
    if( (y{numel(model),1}(1,nAmos) >= 0 & dTeste(1,nAmos) >= 0) | (y{numel(model),1}(1,nAmos) < 0 & dTeste(1,nAmos) < 0) ), acertosTeste = acertosTeste + 1; end
end
acertosTeste
acertosTeste / size(dTeste, 2) * 100
epocas

%% PLOTAGEM DO ERRO MÉDIO QUADRÁTICO

plot(EMQ, 'r-', 'LineWidth', 2);
set(gca, 'xtick', 1 : floor( epocas / 10 ) : epocas, 'ytick', 0: max(EMQ) / 10 : max(EMQ), 'FontSize', 12, 'lineWidth',1.7);
title('Erro Quadrático Médio x Época de Treinamento')
xlabel('Época de Treinamento'), ylabel('Erro Quadrático Médio'), grid
