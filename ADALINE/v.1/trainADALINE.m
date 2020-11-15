function [w, w_0, epcs, erro] = trainADALINE(x, d, eta, sigma)
% Função de treino de uma rede ADALINE de camada única e neurônio único
% 
% ELTON SOARES SILVA - 41711ETE010.
% SAMUEL D. LIMA     - 41621ETE011
% 
% Recebe:
% x:     A matriz de dados de entrada x, onde cada linha contém os dados de cada
%        entrada x1, x2, ... , xn, e cada coluna é uma amostra completa. 
%        Obs.: a 1° linha sempre é uma linha de "uns" devido a questões de
%        implementação.
% 
% d:     é uma matriz linha, contendo a saída desejada para cada amostra de entradas em x    
% 
% eta:   taxa de aprendizagem da rede neural, variando entre 0 e 1.
% 
% sigma: é o patamar aceitável de diferença de erro quadrático médio entre
% duas sucessivas épocas de treinamento
% 
% Devolve:
% w:    matriz linha contendo o valor do limiar seguido dos pesos
%       sinápticos encontrados após o processo de aprendizagem.
% 
% w_0:  Primeiro conjunto de limiar e pesos obtido, gerados aleatoriamente.
% 
% epcs: N° de épocas/iterações gastas p/ treinar a rede.
% 
% erro: matriz linha contendo o valor do erro médio quadrático a cada
% época

   [L, C]        = size(x);      % Dimensões da matriz de dados de entradas
   w             = rand(L, 1);   % inicializando vetor pesos/limiar c/ valores aleatórios como uma matriz linha
   w_0           = w;            % Guarda os valores de pesos e limiar gerados aleatoriamente
   epcs          = 0;            % inicializando contador de épocas                      
   erroAtual     = 123456789;    % inicialização de um valor qualquer grande para o erro quadrático médio atual
   erroAnterior  = 0;            % inicialização da variável que receberá o erro quadrático médio da época anterior
   
%  Executa o while enquanto o patamar de diferença de erro entre épocas não for obtido
%  ou enquanto o n° de épocas for menor ou igual a 20000
   while ( (abs(erroAtual - erroAnterior) >= sigma) && epcs <= 20000 )
%      Executa o for para cada amostra, obtendo o potencial de ativação
%      para cada uma, como sendo o produto da matriz transposta (matriz linha)
%      de pesos e limiar pela matriz de amostras (matriz coluna)
%      Depois, calcula o vetor w usando a taxa de aprendizado, o erro e as
%      entradas de cada amostra
       
       erroAnterior = erroAtual;
        
       for k = 1:C
           u = w' * x(:,k);        
           w = w + eta * ( d(1,k) - u ) * x(:,k);
       end
       
%      Usa a função erroQuad para calcular o erro quadrático médio na época
%      atual
       erroAtual = erroQuad();

%      incrementa o contador de épocas
       epcs = epcs + 1;
       
%      armazena na matriz linha erros, os sucessivos valores de erros médio
%      quadráticos
       erro(1,epcs) = erroAtual;
       
   end
   
   
   %% Função aninhada que calcula o erro quadrático médio
   
   % Inicializa a variável do erro quadrático em 0
   % calcula o erro usando cada amostra
   % encerra o cálculo do erro ao dividir pelo número de amostras
   % o número de amostras de treinamento é o mesmo que o n° de colunas C
    function MSE = erroQuad()
        MSE = 0;
        for kk = 1:C
            u = w' * x(:,kk);
            MSE = MSE + ( d(1,kk) - u )^2;
        end
        MSE = MSE / C;        
    end

end
