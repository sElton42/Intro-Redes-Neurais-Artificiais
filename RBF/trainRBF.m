function [w, w_0, epcs, EMQ ,vari] = trainRBF(x, d, eta, sigma, maxEpcs, model)

    [L, C]        = size(x);      % dimensões da matriz de amostras
    epcs          = 0;            % inicializando contador de épocas
    erroAtual     = 123456789;    % inicialização de um valor qualquer grande para o erro quadrático médio atual
    erroAnterior  = 0;            % inicialização da variável que receberá o erro quadrático médio da época anterior
    grupoAtual    = 0;            % Inicialização do vetor contendo a qual cluster pertence cada amostra na iteração atual
    grupoAnterior = 1;            % Inicialização do vetor contendo a qual cluster pertence cada amostra na iteração anterior
    nAmPG         = 0;            % Inicia o contador de amostras por grupos
    
%   inicialização do contador de número de elementos em cada grupo
    for j = 1:model(1)
        nElemen{j,1} = 0;
    end
    
%   inicialização do erro médio quadrático
    EMQ = 0;
    
    %   PEGANDO AMOSTRAS COM PRESENÇA DE RADIAÇÃO PARA O K-MEANS
    k = 1;
    for j = 1:C
        if( d(model(2), j) == 1 )
            xk(:, k) = x(:, j);
            k = k + 1;
        end
    end
    
%   Dimensões da matriz de amostras
    [Lk, Ck] = size(xk);
    
%   ------------- Inicialização Pesos -------------
    w{2, 1} = rand( model(2), model(1) + 1 ); % + 1 devido ao bias de cada neurônio da camada de saída
 
    w{1, 1} = zeros( model(1) , L-1 );
    for nNeuron = 1 : model(1)
%           cada neurônio tem o centro definido pelas coordenadas na linha
            w{1, 1}( nNeuron , : ) = xk( 2:L, nNeuron);
    end
    w_0 = w;
%   --------------------------
    
%   ************ K-MEANS - TREINAMENTO CAMADA OCULTA ************

%   Executa até não ter diferença no grupo de cada amostra
    while( sum( (grupoAtual - grupoAnterior).^2 ) ~= 0 )
        
%       Armazena os grupos de cada amostra da época anterior em
%       grupoAnterior para efeitos de controle do código
        grupoAnterior = grupoAtual;
        
%       Calcula a distância euclidiana entre cada amostra e os pesos       
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
%       para cada peso, pega as amostras pertencetes a seu grupo
%       soma as coordenadas de cada dimensão
%       divide a soma de cada coordenada pelo número de amostras dentro do
%       grupo
%       
        for nPeso_3 = 1:model(1)
            for nAmostra_3 = 1:Ck
                if( grupos(1, nAmostra_3) == nPeso_3 ) % se o elemento do grupo for igual ao número do peso:
                    soma{nPeso_3, 1}(:, nAmostra_3) = xk(2:L, nAmostra_3); % armazena em soma todos os elementos pertencentes ao grupo do peso nPeso_3
                    nAmPG = nAmPG + 1; % armazena o número de amostras por grupo
                end
            end
            soma = sum( soma{nPeso_3, 1}' ); % <------- debug
            soma = soma / nAmPG;
            w{1,1}(nPeso_3, :) = soma; % atualiza os centros
            nAmPG = 0;
            clear soma
        end
%   **************************
    end
    
%   Após definir os centros, calcula a variância p/ cada neurônio após
%   fazer a diferença entre cada amostra e o centro para cada neurônio
    for nPeso_4 = 1 : model(1)
        for nAmostra_4 = 1:Ck
              if( grupos(1, nAmostra_4) == nPeso_4 ) % se o elemento do grupo for igual ao número do peso:
                  nElemen{nPeso_4, 1} = nElemen{nPeso_4, 1} + 1; % armazena o número de amostras por grupo
                  difAMC(:, nElemen{nPeso_4,1}) = ( xk(2:L, nAmostra_4) - w{1,1}(nPeso_4, :)' ).^2;
              end
        end
          som = sum(difAMC);
          som2 = sum(som);
          vari(nPeso_4, 1) = 1/nElemen{nPeso_4, 1} * som2
    end

%   Treinamento da camada de saída
    while( abs(erroAtual - erroAnterior) >= sigma && epcs <= maxEpcs )
        
        erroAnterior = erroAtual;
        
        
%       APLICAÇÃO DAS ENTRADAS NA CAMADA OCULTA
        for nPeso_5 = 1:model(1)
            for nAmostra_5 = 1:C
%               aplica a função de ativação para cada amostra em cada
%               neurônio.
                gu{nPeso_5,1}(1, nAmostra_5) = exp( (-1) * norm( x(2:L, nAmostra_5) - w{1,1}(nPeso_5, :)' )^2 / ( 2*vari(nPeso_5, 1) ) );
            end
        end
        
%       APLICAÇÃO DAS SAÍDAS DA CAMADA OCULTA NA CAMADA DE SAÍDA
       for nAmostra_6 = 1:C
           for nPeso_6 = 1:model(1)
               vetorAmostras(nPeso_6, nAmostra_6) = gu{nPeso_6, 1}(1,nAmostra_6);
           end
       end
       
       vetorAmostras_bias = [ (-1)*ones( 1, size(vetorAmostras, 2) ) ; vetorAmostras];
       
%      saídas da camada de saída
       y = w{2,1} * vetorAmostras_bias;
       
%      BACKPROPAGATION
       erro = d - y;
       
       for nAmostra_7 = 1:C
           w{2,1} = w{2,1} + eta .* erro(1,nAmostra_7) .* vetorAmostras_bias(:, nAmostra_7)';
       end
       epcs = epcs + 1;
       erroSoma = sum(0.5 * erro.^2);
       EMQ(epcs) = 1/C * erroSoma;
       erroAtual = EMQ(epcs);
    end
    
%   CHECAGEM DE ACERTOS DO TREINAMENTO
    acertos = 0;
    for nAmos = 1:C
        if( y(1,nAmos)*d(1,nAmos) > 0 ), acertos = acertos + 1; end
    end
    plot(EMQ, '-r', 'LineWidth', 2)
    title('Gráfico de Erro','LineWidth', 2)
    legend('Érro Médio Quadrático')
    xlabel(['Épocas [n = ' num2str(epcs) ']'])
    ylabel('EMQ')
    
end