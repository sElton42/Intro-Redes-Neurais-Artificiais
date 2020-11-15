%% FUN��O TREINAMENTO DO PERCEPTRON

function [W,epoca,O,PErro,y] = adaline_Train(X,d,W,N,E,O) 
[nl,nc] = size(X); %Dimens�es da Matriz de dados (X)
Eqm_Anterior = 0;  %Erro quadr�tico m�dio - Anterior
cond = 1;  %Vari�vel condicional
epoca = 0; %Inicializa��o da contagem de �pocas

%% - LOOP
while cond == 1 % Expressa a condi��o de minimiza��o do EQM
    for a=1:nl
    u = sum((W.*X(a,:)) + O,'all'); %Somat�rio u=X(k)*W(k)+O
    W = W + N*(d(a)-u)*X(a,:);      %Ajuste dos Pesos (W)
    O = O + N*(d(a)-u);             %Ajuste do Limiar de Ativa��o
    erro(a) =  (d(a)-u)^2;
    
        if u >= 0
            y(a) = 1;
            else
            y(a) = -1;
        end
    end
    
    Eqm_Atual = (1/nc)*sum(erro,'all');
    
    
    if abs(Eqm_Atual-Eqm_Anterior)<E
        cond = 0;
    else
        Eqm_Anterior = Eqm_Atual;
    end
    epoca = epoca + 1;
    PErro(epoca) = sum((erro.*2)/nc);
end
% PLOTS
figure
p = plot([1:epoca],PErro,'r')
p(1).LineWidth = 2;
grid minor
title('Gr�fico de Erro (Treinamento)','Color','black')
xlabel('�pocas [n]','FontSize',12,'FontWeight','bold','Color','black')
ylabel('Erro [(Eqm�/p)(n)]','FontSize',12,'FontWeight','bold','Color','black')
legend('Eqm�/EP')
return
end