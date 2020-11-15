%% - PRÁTICA 02 RNA ADALINE
%Nomes: Samuel D. Lima - 41621ETE011
%       Elton S. Silva - 41711ETE010
%% Inicialização
clear all
close all
clc

%% IMPORTAR DADOS
X = csvread('DATA Train'); %Importa Dados para Treinamento
d = csvread('DATA d');     %Importa os resultados desejados
Xf = csvread('DATA Clas'); %Importa os dados para classificação
T_1a5 = csvread('T_1a5');  %Importa Dados do Treinamento

%% VARIÁVEIS INICIAIS
W = rand(1,4); %Vetor (linha) de Pesos
N = 0.0025;    %Taxa de Aprendizado
E = 1e-6;      %Precisão da Convergência
O = -1;
%% FUNÇÃO DE TREINAMENTO (GERA os DADOS da QUESTÃO 2)
% for a=1:5
%     W = rand(1,4);
%     O = rand(1);
%     [Watual,epoca,Oatual,PErro,y] = adaline_Train(X,d,W,N,E,O);
%     T(a,:) = [O W Oatual Watual epoca]
% end
%% QUESTÃO 02

for a=1:2
[Watual,epoca,Oatual,PErro,y] = adaline_Train(X,d,T_1a5(a,[2:5]),N,E,T_1a5(a,1))
figure
p = plot([1:epoca],PErro,'r')
p(1).LineWidth = 2;
grid minor
title('Gráfico de Erro (Treinamento)','Color','black')
xlabel('Épocas [n]','FontSize',12,'FontWeight','bold','Color','black')
ylabel('Erro [(Eqm²/p)(n)]','FontSize',12,'FontWeight','bold','Color','black')
legend('Eqm²/p')
hold on
end



%% FUNÇÃO DE CLASSIFICAÇÃO

    for a=1:5
        resp(:,a) = adaline_Func(Xf,T_1a5(a,[7:10]),T_1a5(a,6));
    end
csvwrite('Resp_T1a5',resp)

%% RESULTADOS
figure
    stem((resp),'b')
    p(1).LineWidth = 2;
    grid minor
    axis([0 length(resp)+1 -2  2])
    title('Classificação','Color','black')
    xlabel('d[n]','FontSize',12,'FontWeight','bold','Color','black')
    ylabel('y[n]','FontSize',12,'FontWeight','bold','Color','black')
