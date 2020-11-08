%% PRÁTICA 01 - PERCEPTRON
%Nomes: Samuel D. Lima - Elton S. Silva

%% Inicialização
%OBS: Processo Necessário para limpar todas as variáveis e fechar processos
%abertos!
clear all
close all
clc

%% UPAR DADOS
X = [-0.6508,-1.4492,2.0850,0.2626,0.6418,0.2569,1.1155,0.0914,0.0121,-0.0429,0.4340,0.2735,0.4839,0.4089,1.4391,-0.9115,0.3654,0.2144,0.2013,0.6483,-0.1147,-0.7970,-1.0625,0.5307,-1.2200,0.3957,-0.1013,2.4482,2.0149,0.2012;
     0.1097,0.8896,0.6876,1.1476,1.0234,0.6730,0.6043,0.3399,0.5256,0.4660,0.6870,1.0287,0.4851,-0.1267,0.1614,-0.1973,1.0475,0.7515,1.0014,0.2183,0.2242,0.8795,0.6366,0.1285,0.7777,0.1076,0.5989,0.9455,0.6192,0.2611;
     4.0009,4.4005,12.0710,7.7985,7.0427,8.3265,7.4446,7.0677,4.6316,5.4323,8.2287,7.1934,7.4850,5.5019,8.5843,2.1962,7.4858,7.1699,6.5489,5.8991,7.2435,3.8762,2.4707,5.6883,1.7252,5.6623,7.1812,11.2095,10.9263,5.4631]; 
     % Amostras de Treinamento d = ; % Resultados (Target)
d = [-1.0000,-1.0000,-1.0000,1.0000,1.0000,-1.0000,1.0000,-1.0000,1.0000,1.0000,-1.0000,1.0000,-1.0000,-1.0000,-1.0000,-1.0000,1.0000,1.0000,1.0000,1.0000,-1.0000,1.0000,1.0000,1.0000,1.0000,-1.0000,-1.0000,1.0000,-1.0000,1.0000]; % Amostras para classificaÃ§Ã£o
Xf = [-0.356500000000000,-0.784200000000000,0.301200000000000,0.775700000000000,0.157000000000000,-0.701400000000000,0.374800000000000,-0.692000000000000,-1.39700000000000,-1.88420000000000;0.0620000000000000,1.12670000000000,0.561100000000000,1.06480000000000,0.802800000000000,1.03160000000000,0.153600000000000,0.940400000000000,0.714100000000000,-0.280500000000000;5.98910000000000,5.59120000000000,5.82340000000000,8.06770000000000,6.30400000000000,3.60050000000000,6.15370000000000,4.40580000000000,4.92630000000000,1.25480000000000];

%% Ex1 (Os dados do exercício 1 serão colocados em T(1 ao 5))
n = 0.01;  % Define a taxa de aprendizagem

for si=1:5
    w = rand(1,3); %Gera valore pseudo aleatórios para os pesos
    b=rand(1); %Gera valores pseudo aleatórios para o bias
    [pesos,bias,epoca]=w_perceptron(X,d,w,n,b); % Chama a função que retorna os pesos do treinamento
    T(si,:)=[b,w,bias,pesos,epoca];  % Armazena os dados
end



%% Generalização
for z=1:length(Xf)
    resp(z) = funcao_perceptron(Xf(:,z),T(2,(6:8)),T(2,(5)));
end
resp % Variável que contém as Classificações
% plot(resp,"o")
% axis([0 length(Xf) -2 2])
% title("Resultado RNA "); xlabel("Amostra"); ylabel("C1               |                C2")
% grid on
%%R
for aa=1:length(X)
    if(d(aa)==-1)
        d1(aa)=0;
    else
        d1(aa)=1;
    end
end
for aa=1:length(resp)
    if(d(aa)==-1)
        d2(aa)=0;
    else
        d2(aa)=1;
    end
end

%%PLOTS
a = -5:5; % Limites(x)
b = a; % Limites(y)
[cx,cy] = meshgrid(a,b);
z = -(a.*pesos(1)+b'.*pesos(2) + bias)./pesos(3);
figure
surf(cx,cy,z)
hold on
plot3(Xf(1,:),Xf(2,:),Xf(3,:),'o')
title('Dados para a Classificação')
grid on

% %% Plot dos dados de entrada
% plotpv(X,d1)
% plotpc(pesos,bias)
% title('Dados de Treinamento')
% 
% %% Plot dos dados analisados (Final)
% figure
% plotpv(Xf,d2)
% plotpc(pesos,bias)
% title('Dados para a Classificação')

%% Condição de Menor época para os treinamentos escolhidos (CURIOSIDADE)
% n = 0.01;
% b = [-5 -2 0 2 5]; %Bias defino estÃ¡ticamente para o teste
% cond=1;
% np=15; %Variável principal - Define a menor Ã©poca para cada Bias escolhido 
%        %O menor número de épocas define os valores aleatórios de pesos para
%        %a convergência
% T = zeros(5,9);
% while cond == 1
% for si=1:5
%     w = rand(1,3);
%     [pesos,bias,epoca]=w_perceptron(X,d,w,n,b(si));
%     if (epoca < np)
%     T(si,:)=[b(si),w,bias,pesos,epoca]; 
%     c(si) = 1;
%     if sum(c)==5 cond =0; end
%     end
%     
% end
% Text = ['bi w1 w2 w3 ba wf1 wf2 wf3 ep'] %Bias Incial,w1,w2,w3,Bias Final,wf1,wf2,wf3,epoca
% T % Mostra a Matriz T
% end








