%% - PR�TICA 02 RNA ADALINE
%Nomes: Samuel D. Lima - 41621ETE011
%       Elton S. Silva - 41711ETE010
%% Inicializa��o
clear all
close all
clc

%% IMPORTAR DADOS
X = csvread('DATA Train'); %Importa Dados para Treinamento
d = csvread('DATA d');     %Importa os resultados desejados
Xf = csvread('DATA Clas'); %Importa os dados para classifica��o
T_1a5 = csvread('T_1a5');  %Importa Dados do Treinamento

%% VARI�VEIS INICIAIS
W = rand(1,4); %Vetor (linha) de Pesos
N = 0.0025;    %Taxa de Aprendizado
E = 1e-6;      %Precis�o da Converg�ncia
O = -1;
%% FUN��O DE TREINAMENTO (GERA os DADOS da QUEST�O 2)
% for a=1:5
%     W = rand(1,4);
%     O = rand(1);
%     [Watual,epoca,Oatual,PErro,y] = adaline_Train(X,d,W,N,E,O);
%     T(a,:) = [O W Oatual Watual epoca]
% end
%% QUEST�O 02

for a=1:2
[Watual,epoca,Oatual,PErro,y] = adaline_Train(X,d,T_1a5(a,[2:5]),N,E,T_1a5(a,1))
figure
p = plot([1:epoca],PErro,'r')
p(1).LineWidth = 2;
grid minor
title('Gr�fico de Erro (Treinamento)','Color','black')
xlabel('�pocas [n]','FontSize',12,'FontWeight','bold','Color','black')
ylabel('Erro [(Eqm�/p)(n)]','FontSize',12,'FontWeight','bold','Color','black')
legend('Eqm�/p')
hold on
end



%% FUN��O DE CLASSIFICA��O

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
    title('Classifica��o','Color','black')
    xlabel('d[n]','FontSize',12,'FontWeight','bold','Color','black')
    ylabel('y[n]','FontSize',12,'FontWeight','bold','Color','black')
