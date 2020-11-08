%% Fun��o PERCEPTRON

function [y] = funcao_perceptron(X,W,bias)
% X -> Vetor (Coluna) define as Entradas (INPUTS)
% W -> Vetor (Linha) define os pesos sin�pticos
% bias -> vari�vel num�rica, define o ponto que intercepta o eixo de ajuste!
 somatorio = sum(X.* transpose(W),'all') + bias; % y'(k) = x(k)i*W(k)i + b(k);
     % Fun��o de Ativa��o
     if somatorio >= 0
                y = 1;
     else
                y = -1;
     end
     return
end
