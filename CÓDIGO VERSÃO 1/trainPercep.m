function [w, w_0, epcs] = trainPercep(x, d, eta)
% Fun��o de treino de uma rede Perceptron de camada �nica e neur�nio �nico
% 
% ELTON SOARES SILVA - 41711ETE010.
% SAMUEL D. LIMA     - 41621ETE011
% 
% Recebe:
% x:    A matriz de dados de entrada x, onde cada linha cont�m os dados de cada
%       entrada x1, x2, ... , xn, e cada coluna � uma amostra completa. 
%       Obs.: a 1� linha sempre � uma linha de "uns" devido a quest�es de
%       implementa��o.
% 
% d:    � uma matriz linha, contendo a sa�da desejada para cada amostra de entradas em x    
% 
% eta:  taxa de aprendizagem da rede neural, variando entre 0 e 1.
% 
% Devolve:
% w:    matriz linha contendo o valor do limiar seguido dos pesos
%       sin�pticos encontrados ap�s o processo de aprendizagem.
% 
% w_0:  Primeiro conjunto de limiar e pesos obtido, gerados aleatoriamente.
% 
% epcs: N� de �pocas/itera��es gastas p/ treinar a rede.

   [L, C]     = size(x);      % Dimens�es da matriz de dados de entradas              
   w          = rand(L, 1);   % inicializando vetor pesos/limiar c/ valores aleat�rios como uma matriz linha
   w_0        = w;            % Guarda os valores de pesos e limiar gerados aleatoriamente
   epcs       = 0;            % inicializando contador de �pocas                      
   erro       = ones(1,C);    % Inicializa a matriz para armazenar a diferen�a entre as sa�das desejadas e a obtidas
   
%  Executa o while enquanto houver diferen�a entre sa�da desejada e obtida
%  ou enquanto o n� de �pocas for menor ou igual a 20000
   while (sum(abs(erro)) && epcs <= 20000)
%      Executa o for para cada amostra, obtendo o potencial de ativa��o
%      para cada uma como sendo o produto da matriz transposta (matriz linha)
%      de pesos e limiar pela matriz de amostras (matriz coluna)
%      Depois, usa a fun��o degrau para obter a classifica��o de cada
%      amostra, fazendo a diferen�a entre desejado e obtido e, caso seja
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


%% Fun��o Degrau bipolar
% Retorna   1, se u >= 0
% Retorna  -1, se u < 0
function y = degrau(u)
        if ( u >= 0 ), y = 1; else, y = -1; end
end