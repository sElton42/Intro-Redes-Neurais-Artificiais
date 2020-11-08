function [w, w_0, epcs] = trainPercep(x, d, eta)
% Função de treino de uma rede Perceptron de camada única e neurônio único
% 
% ELTON SOARES SILVA - 41711ETE010.
% SAMUEL D. LIMA     - 41621ETE011
% 
% Recebe:
% x:    A matriz de dados de entrada x, onde cada linha contém os dados de cada
%       entrada x1, x2, ... , xn, e cada coluna é uma amostra completa. 
%       Obs.: a 1° linha sempre é uma linha de "uns" devido a questões de
%       implementação.
% 
% d:    é uma matriz linha, contendo a saída desejada para cada amostra de entradas em x    
% 
% eta:  taxa de aprendizagem da rede neural, variando entre 0 e 1.
% 
% Devolve:
% w:    matriz linha contendo o valor do limiar seguido dos pesos
%       sinápticos encontrados após o processo de aprendizagem.
% 
% w_0:  Primeiro conjunto de limiar e pesos obtido, gerados aleatoriamente.
% 
% epcs: N° de épocas/iterações gastas p/ treinar a rede.

   [L, C]     = size(x);      % Dimensões da matriz de dados de entradas              
   w          = rand(L, 1);   % inicializando vetor pesos/limiar c/ valores aleatórios como uma matriz linha
   w_0        = w;            % Guarda os valores de pesos e limiar gerados aleatoriamente
   epcs       = 0;            % inicializando contador de épocas                      
   erro       = ones(1,C);    % Inicializa a matriz para armazenar a diferença entre as saídas desejadas e a obtidas
   
%  Executa o while enquanto houver diferença entre saída desejada e obtida
%  ou enquanto o n° de épocas for menor ou igual a 20000
   while (sum(abs(erro)) && epcs <= 20000)
%      Executa o for para cada amostra, obtendo o potencial de ativação
%      para cada uma como sendo o produto da matriz transposta (matriz linha)
%      de pesos e limiar pela matriz de amostras (matriz coluna)
%      Depois, usa a função degrau para obter a classificação de cada
%      amostra, fazendo a diferença entre desejado e obtido e, caso seja
%      diferente, recalcula os pesos e limiar.
       for k = 1:C
           u = w' * x(:,k);        
           y = degrau(u);
           erro(1,k) = d(1,k) - y;
           w = w + eta * erro(1,k) * x(:,k);
       end
       
       epcs = epcs + 1;
       
   end
   
end


%% Função Degrau bipolar
% Retorna   1, se u >= 0
% Retorna  -1, se u < 0
function y = degrau(u)
        if ( u >= 0 ), y = 1; else, y = -1; end
end