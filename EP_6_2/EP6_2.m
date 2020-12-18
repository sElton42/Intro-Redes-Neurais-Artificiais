% ELTON SOARES SILVA    - 41711ETE010
% SAMUEL DAVI DE LIMA   - 41621ETE011

clear, clc, close all

%% Import dos Dados

% Dados de Treino
L = 150; C = 4;
numSaidas = 1;
filenameTrain = 'dadosTreinamento.csv';

[X, d] = dataImport(filenameTrain, L, C, numSaidas);

% Dados Novos
L = 15; C = 4;
numSaidas = 1;
filenameNew = 'dadosNovos.csv';

[Xnew, dnew] = dataImport(filenameNew, L, C, numSaidas);


%% TREINO DA REDE

% Inicializa��es
eta = 0.01;     % Taxa de treinamento
sigma = 1E-7;   % Precis�o requerida
maxEpcs = 3000; % N�mero m�ximo de �pocas permitidas
model = [15 1]; % N�mero de neur�nios por camada

[w, w0, epcs, EMQ, vari] = trainRBF(X, d, eta, sigma, maxEpcs, model);

%% OPERA��O DA REDE NO CONJUNTO DE TESTES

% APLICA��O DAS ENTRADAS NA CAMADA OCULTA
for nPeso = 1:model(1)
    for nAmost = 1 : size(Xnew,2)
        % aplica a fun��o de ativa��o para cada amostra em cada
        % neur�nio.
        gu{nPeso,1}(1, nAmost) = exp( (-1) * norm( Xnew(2:size(Xnew,1), nAmost) - w{1,1}(nPeso, :)' )^2 / ( 2*vari(nPeso, 1) ) );
    end
end

% APLICA��O DAS SA�DAS DA CAMADA OCULTA NA CAMADA DE SA�DA
for nAmost2 = 1:size(Xnew,2)
    for nPeso2 = 1:model(1)
        vetorAmostras(nPeso2, nAmost2) = gu{nPeso2, 1}(1,nAmost2);
    end
end

vetorAmostras_bias = [ (-1)*ones( 1, size(vetorAmostras, 2) ) ; vetorAmostras];

saidaTeste = w{2,1} * vetorAmostras_bias;

% arredonda a sa�da e converte para texto
% para ficar f�cil de copiar para a tabela do relat�rio no word
y = round(saidaTeste, 4);
yText = arrayfun(@num2str, y, 'un', 0);
yText = yText';

%% ERRO RELATIVO M�DIO E C�LCULO DA VARI�NCIA

erroRel = abs(saidaTeste - dnew);
erroRelMed = sum(erroRel) / numel(erroRel) * 100
variancia = var(erroRel) * 100

%% PLOTAGEM DO ERRO M�DIO QUADR�TICO

plot(EMQ, 'r-', 'LineWidth', 2);
set(gca, 'xtick', 1 : floor( epcs / 10 ) : epcs, 'ytick', 0: max(EMQ) / 10 : max(EMQ), 'FontSize', 12, 'lineWidth',1.7);
title('Erro Quadr�tico M�dio x �poca de Treinamento')
xlabel('�poca de Treinamento'), ylabel('Erro Quadr�tico M�dio'), grid
