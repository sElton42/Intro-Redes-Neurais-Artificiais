function [w, w_0, epcs, erro] = trainADALINE(x, d, eta, sigma)
% Fun��o de treino de uma rede ADALINE de camada �nica e neur�nio �nico
% 
% ELTON SOARES SILVA - 41711ETE010.
% SAMUEL D. LIMA     - 41621ETE011
% 
% Recebe:
% x:     A matriz de dados de entrada x, onde cada linha cont�m os dados de cada
%        entrada x1, x2, ... , xn, e cada coluna � uma amostra completa. 
%        Obs.: a 1� linha sempre � uma linha de "uns" devido a quest�es de
%        implementa��o.
% 
% d:     � uma matriz linha, contendo a sa�da desejada para cada amostra de entradas em x    
% 
% eta:   taxa de aprendizagem da rede neural, variando entre 0 e 1.
% 
% sigma: � o patamar aceit�vel de diferen�a de erro quadr�tico m�dio entre
% duas sucessivas �pocas de treinamento
% 
% Devolve:
% w:    matriz linha contendo o valor do limiar seguido dos pesos
%       sin�pticos encontrados ap�s o processo de aprendizagem.
% 
% w_0:  Primeiro conjunto de limiar e pesos obtido, gerados aleatoriamente.
% 
% epcs: N� de �pocas/itera��es gastas p/ treinar a rede.
% 
% erro: matriz linha contendo o valor do erro m�dio quadr�tico a cada
% �poca

   [L, C]        = size(x);      % Dimens�es da matriz de dados de entradas
   w             = rand(L, 1);   % inicializando vetor pesos/limiar c/ valores aleat�rios como uma matriz linha
   w_0           = w;            % Guarda os valores de pesos e limiar gerados aleatoriamente
   epcs          = 0;            % inicializando contador de �pocas                      
   erroAtual     = 123456789;    % inicializa��o de um valor qualquer grande para o erro quadr�tico m�dio atual
   erroAnterior  = 0;            % inicializa��o da vari�vel que receber� o erro quadr�tico m�dio da �poca anterior
   
%  Executa o while enquanto o patamar de diferen�a de erro entre �pocas n�o for obtido
%  ou enquanto o n� de �pocas for menor ou igual a 20000
   while ( (abs(erroAtual - erroAnterior) >= sigma) && epcs <= 20000 )
%      Executa o for para cada amostra, obtendo o potencial de ativa��o
%      para cada uma, como sendo o produto da matriz transposta (matriz linha)
%      de pesos e limiar pela matriz de amostras (matriz coluna)
%      Depois, calcula o vetor w usando a taxa de aprendizado, o erro e as
%      entradas de cada amostra
       
       erroAnterior = erroAtual;
        
       for k = 1:C
           u = w' * x(:,k);        
           w = w + eta * ( d(1,k) - u ) * x(:,k);
       end
       
%      Usa a fun��o erroQuad para calcular o erro quadr�tico m�dio na �poca
%      atual
       erroAtual = erroQuad();

%      incrementa o contador de �pocas
       epcs = epcs + 1;
       
%      armazena na matriz linha erros, os sucessivos valores de erros m�dio
%      quadr�ticos
       erro(1,epcs) = erroAtual;
       
   end
   
   
   %% Fun��o aninhada que calcula o erro quadr�tico m�dio
   
   % Inicializa a vari�vel do erro quadr�tico em 0
   % calcula o erro usando cada amostra
   % encerra o c�lculo do erro ao dividir pelo n�mero de amostras
   % o n�mero de amostras de treinamento � o mesmo que o n� de colunas C
    function MSE = erroQuad()
        MSE = 0;
        for kk = 1:C
            u = w' * x(:,kk);
            MSE = MSE + ( d(1,kk) - u )^2;
        end
        MSE = MSE / C;        
    end

end
