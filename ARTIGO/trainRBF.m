function [w, w_0, epcs, EMQ, accuracyTrain, vari] = trainRBF(x, d, eta, sigma, maxEpcs, model)

%   ------------- Inicialização Variáveis da rede -------------------------
    [L, C]        = size(x);    % dimensões da matriz de amostras
    epcs          = 0;          % inicializando contador de épocas
    erroAtual     = 123;        % inicialização de um valor qualquer grande para o erro quadrático médio atual
    erroAnterior  = 0;          % inicialização da variável que receberá o erro quadrático médio da época anterior
    grupoAtual    = 0;          % inicializa a variável que armazena quais grupos cada amostra pertence no k-means
    grupoAnterior = 1;          % inicializa a variável que armazena quais grupos cada amostra pertence no k-means da ITERAÇÃO ANTERIOR
    nAmPG         = 0;          % inicializa a variável que conta o n° de amostras por grupo
    
%   inicialização do contador do número de elementos
    for j = 1:model(1)
        nElemen{j,1} = 0;
    end
    
%   inicialização do erro médio quadrático
    EMQ = 0;
    
%   PEGANDO AMOSTRAS CLASSIFICADAS COMO UM DOS PADRÕES (1 ou -1) p/
%   CLUSTERIZAÇÃO
    k = 1;
    for j = 1:C
        if( d(model(2), j) == 1 )
            xk(:, k) = x(:, j);
            k = k + 1;
        end
    end
    
%   Dimensões do grupo de amostras
    [Lk, Ck] = size(xk);
    
%   -----------------------------------------------------------------------
    
%   ------------- Inicialização Pesos -------------------------------------

%   inicializa pesos entre camada oculta e de saída
%   + 1 devido ao bias de cada neurônio da camada de saída
    w{2, 1} = rand( model(2), model(1) + 1 );
 
%   inicializa centros de cada neurônio da camada escondida como sendo as n
%	primeiras amostras do grupo de amostras de um só padrão, xk
    w{1, 1} = zeros( model(1) , L-1 );
    for nNeuron = 1 : model(1)
%           cada neurônio tem o centro definido pelas coordenadas na linha
            w{1, 1}( nNeuron , : ) = xk( 2:L, nNeuron);
    end
    w_0 = w; % guarda os pesos de inicialização
%   -----------------------------------------------------------------------

%   ------------- Algoritmo K-means  --------------------------------------
%   Define os centros e variâncias de cada neurônio da camada oculta

%   executa enquanto houver mudanças de amostras entre os grupos
    while( sum( (grupoAtual - grupoAnterior).^2 ) ~= 0 )
        
%       Armazena os grupos de cada amostra da época anterior em
%       grupoAnterior para efeitos de controle do código
        grupoAnterior = grupoAtual;
        
%       Calcula a distância euclidiana entre cada amostra e os centros de
%       cada neurônio
        for nPeso = 1:model(1)
            for nAmostra = 1:Ck
                for nCoordenada = 1 : Lk-1
                    diferencas(nCoordenada, nAmostra) = ( w{1,1}(nPeso, nCoordenada) - xk(nCoordenada + 1, nAmostra) ).^2;
                end
                somaDiferencas = sum(diferencas);

                % Distâncias de cada amostra até o centro. Cada linha é um
                % neurônio
                distancias{nPeso,1}(1, nAmostra) = sqrt( somaDiferencas(:,nAmostra) );
            end
        end
        

%       armazena em grupo, de acordo com o n° da amostra,
%       o grupo a qual ela pertence, que é aquele em que ela está
%       mais próxima do peso (centro)
%       depois, armazena os valores desse grupo em grupoAtual apenas
%       para controle do algoritmo
        for nAmostra_2 = 1:Ck
            for nPeso_2 = 1:model(1)
                dists_amostra(1, nPeso_2) = distancias{nPeso_2,1}(1, nAmostra_2); 
            end
            [menorValor, idx] = min(dists_amostra); % pega o menor valor de distância
            grupos(1, nAmostra_2) = idx;
            grupoAtual = grupos;
        end
        
%       Realiza a atualização do peso (centro)
%       Para cada peso, pega as amostras pertencetes ao seu grupo,
%       soma as coordenadas de cada dimensão
%       divide a soma de cada coordenada pelo número de amostras dentro do
%       grupo   
        for nPeso_3 = 1:model(1)
            for nAmostra_3 = 1:Ck
                if( grupos(1, nAmostra_3) == nPeso_3 ) % se o elemento do grupo for igual ao número do peso:
                    soma{nPeso_3, 1}(:, nAmostra_3) = xk(2:L, nAmostra_3); % armazena em soma todos os elementos pertencentes ao grupo do peso nPeso_3
                    nAmPG = nAmPG + 1; % armazena o número de amostras por grupo
                end
            end
            soma = sum( soma{nPeso_3, 1}' );
            soma = soma / nAmPG;
            w{1,1}(nPeso_3, :) = soma; % atualiza os centros
            nAmPG = 0;
            clear soma
        end
    end
    
%   Após calcular os centros, calcula a variância de cada neurônio
    for nPeso_4 = 1 : model(1)
        for nAmostra_4 = 1:Ck
              if( grupos(1, nAmostra_4) == nPeso_4 ) % se o elemento do grupo for igual ao número do peso:
                  nElemen{nPeso_4, 1} = nElemen{nPeso_4, 1} + 1; % armazena o número de amostras por grupo
                  difAMC(:, nElemen{nPeso_4,1}) = ( xk(2:L, nAmostra_4) - w{1,1}(nPeso_4, :)' ).^2;
              end
        end
          som = sum(difAMC);
          som2 = sum(som);
          vari(nPeso_4, 1) = 1/nElemen{nPeso_4, 1} * som2;
    end
%   Fim K-means
%   -----------------------------------------------------------------------
  
%   ------------- Treino da Camada de Saída  ------------------------------
    while( abs(erroAtual - erroAnterior) >= sigma && epcs <= maxEpcs )
        
        erroAnterior = erroAtual;
        
        
%       APLICAÇÃO DAS ENTRADAS NA CAMADA OCULTA
        for nPeso_5 = 1:model(1)
            for nAmostra_5 = 1:C
%               aplica a função de ativação (gaussiana) para cada amostra em cada
%               neurônio.
                gu{nPeso_5,1}(1, nAmostra_5) = exp( (-1) * norm( x(2:L, nAmostra_5) - w{1,1}(nPeso_5, :)' )^2 / ( 2*vari(nPeso_5, 1) ) );
            end
        end
        
%      APLICAÇÃO DAS SAÍDAS DA CAMADA OCULTA NA CAMADA DE SAÍDA
%      Criação de um vetor separado para receber a saída dos neurônios da camada oculta 
       for nAmostra_6 = 1:C
           for nPeso_6 = 1:model(1)
               vetorAmostras(nPeso_6, nAmostra_6) = gu{nPeso_6, 1}(1,nAmostra_6);
           end
       end
       
%      Pega a saída dos neurônios da camada oculta e concatena com o
%      multiplicador do bias (-1)
       vetorAmostras_bias = [ (-1)*ones( 1, size(vetorAmostras, 2) ) ; vetorAmostras];
       
%      obtém a saída da camada de saída
       y = w{2,1} * vetorAmostras_bias;
       
%      BACKPROPAGATION

%      Pega o sinal de erro
       erro = d - y;
       
%      Atualiza os pesos usando a regra delta generalizada (treino
%      off-line)
       for nAmostra_7 = 1:C
           w{2,1} = w{2,1} + eta .* erro(1,nAmostra_7) .* vetorAmostras_bias(:, nAmostra_7)';
       end
       
%      Incrementa as épocas, calcula o erro quadrático médio
       epcs = epcs + 1;
%        erroSoma = sum(0.5 * erro.^2);
       erroSoma = sum(sum(0.5 * erro.^2));
       EMQ(epcs) = 1/C * erroSoma;
       erroAtual = EMQ(epcs);
    end
    
%% CHECAGEM DE ACERTOS DO TREINAMENTO
    accuracyTrain = 0;
%     for nAmos = 1:C
%         if( y(1,nAmos)*d(1,nAmos) > 0 ), accuracyTrain = accuracyTrain + 1; end
%     end
end