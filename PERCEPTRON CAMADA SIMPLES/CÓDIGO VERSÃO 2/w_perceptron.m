%% FUN��O TREINAMENTO DO PERCEPTRON

function [W,bias,epoca] = perceptron_Treinamento(X,d,W,taxa_ap,bias) 
% X -> Deve ser um Vetor (Coluna) - Ex: X = [x(1);x(2);x(3),...,x(n)]
% d -> o vetor (Linha) resposta - Ex: d = [r(1) r(2),r(3),...,r(n)]
% W -> Pesos definidos em vetor (Linha) - Ex: W = [w(1),w(2),w(3),...,w(n)]
% taxa_ap -> Taxa de Aprendizagem (0-1)
% bias -> Constante
           %% Treinamento
           %Vari�veis
           epoca = 1; %Contagem das �pocas
           cond = 1; %Vari�vel Condicional
           while cond==1
                for cont=1:length(X)
                    somatorio = sum(X(:,cont).* transpose(W),'all') + bias; % y'(k) = x(k)i*W(k)i + b(k);
                    %% Fun��o de Ativa��o
                    if somatorio >= 0
                        y = 1;
                    else
                        y = -1;
                    end
                    %% C�lculos dos Ajustes
                    erro = d(cont)-y; %e = d(k) - y(k) -> Erro do resultado/sa�da gerada
                    bias = bias + erro; %b(k+1) = b(k) + erro(k)
                    Err(cont)= abs(erro); % -> Condi��o de final de epoca! 
                    %Condiciona se h� erros na epoca atual!
                    
                    if erro == 0 else
                        W = W+transpose(taxa_ap.*erro.*X(:,cont)); %W(k+1) = W(k)+
                    end
                end
                %% VERIFICA��O DE TREINAMENTO
                if (sum(Err,"all")==0)
                    cond = 0;
                end
                epoca = epoca + 1; %Condi��o - Contagem das �pocas
            end
            %% FIM      
  return
end
