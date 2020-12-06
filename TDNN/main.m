%% PRÁTICA 3 - TDNN
% ELTON SOARES SILVA - 41711ETE010
% SAMUEL D. LIMA     - 41621ETE011

%% INICIALIZAÇÃO

clear all
close all
clc

%% DADOS DE TREINO

DADOS = load('X.mat')
DADOS = DADOS.X(:,2);

%% DADOS NOVOS
DADOSNOVOS = csvread('dadosNovos.csv');

% Concatena dados novos com os de treino p/ aplicação da rede
DADOSNOVOS = [DADOS ; DADOSNOVOS];

%% VARIÁVEIS

n = 0.1; % taxa de aprendizagem
ee = 0.5e-6; % precisão

%% TREINAMENTO DA REDE 1 - Np = 5
%  [OUTPUT{1,1}, OUTPUT{1,2}, OUTPUT{1,3}, OUTPUT{1,4}] = pmc_TDNN_M(DADOS,n,ee,5,0.8,10)
%% TREINAMENTO DA REDE 2 - Np = 10
%  [OUTPUT{2,1}, OUTPUT{2,2},OUTPUT{2,3}, OUTPUT{2,4}] = pmc_TDNN_M(DADOS,n,ee,10,0.8,15)
%% TREINAMENTO DA REDE 3 - Np = 15
 [OUTPUT{3,1}, OUTPUT{3,2}, OUTPUT{3,3}, OUTPUT{3,4}] = pmc_TDNN_M(DADOS,n,ee,15,0.8,25)
 
 close all
%% APLICANDO A REDE TREINADA AOS DADOS NOVOS (Trocar o Np para cada rede)


X=[];

for k=1:length(DADOSNOVOS)-15
    X(k,:) = [-1 DADOSNOVOS(k:k+15-1,1)'];
end

d = DADOSNOVOS(15+1:length(DADOSNOVOS))';

dimx = size(X); %Armazena as dimensões da Matriz De dados de Entrada

for a=1:dimx(1) % Percorre todas as Linhas
        
    % -> 1° Camada Escondida
        I1 = X(a,:)*OUTPUT{3,1}';  % somatório: entradas * pesos + bias
        Y1 = (1./(1+exp(-I1)));    % Função de ativação
        Y1 = [-1,Y1];              % concatena multiplicador do bias p/ próxima camada
        
    % -> 2° Camada (camada de saída)
        I2 = Y1*OUTPUT{3,2}';          % somatório: Y1 * pesos + bias
        Y2(a) = (1./(1+exp(-I2)));  % Função de ativação
end

Y2_string = round(Y2(86:105), 4)
Y2_string = arrayfun(@num2str, Y2_string, 'un', 0);
Y2_string = Y2_string';

diferenca = d(86:105) - Y2(86:105)
erroRel = abs(diferenca)  ./ d(86:105)
erroRelMed = sum(erroRel) / numel( Y2(86:105) )
variancia = var( Y2(86:105) )

figure
plot(101:120, d(86:105), '-r', 'LineWidth', 2), hold on
plot(101:120, Y2(86:105), '-b', 'LineWidth', 2)
legend('Curva Desejada','Curva Estimada')
title([' Valores Desejados x Estimados - REDE 3 p=15 N=25 '],'FontSize',12,'FontWeight','bold','Color','black')
xlabel('t','FontSize',12,'FontWeight','bold','Color','black')
ylabel('Resultado y(t)','FontSize',12,'FontWeight','bold','Color','black')

figure
plot(OUTPUT{3,4}, 'LineWidth', 2)
title([' Erro Médio Quadrático por época - REDE 3 p=15 N=25 '],'FontSize',12,'FontWeight','bold','Color','black')
xlabel('Épocas','FontSize',12,'FontWeight','bold','Color','black')
ylabel('MSE','FontSize',12,'FontWeight','bold','Color','black')
