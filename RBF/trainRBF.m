function [w, w_0, epcs, EMQ ,vari] = trainRBF(x, d, eta, sigma, maxEpcs, model)

    [L, C]        = size(x);      % dimens�es da matriz de amostras
    epcs          = 0;            % inicializando contador de �pocas
    erroAtual     = 123456789;    % inicializa��o de um valor qualquer grande para o erro quadr�tico m�dio atual
    erroAnterior  = 0;            % inicializa��o da vari�vel que receber� o erro quadr�tico m�dio da �poca anterior
    grupoAtual    = 0;            % Inicializa��o do vetor contendo a qual cluster pertence cada amostra na itera��o atual
    grupoAnterior = 1;            % Inicializa��o do vetor contendo a qual cluster pertence cada amostra na itera��o anterior
    nAmPG         = 0;            % Inicia o contador de amostras por grupos
    
%   inicializa��o do contador de n�mero de elementos em cada grupo
    for j = 1:model(1)
        nElemen{j,1} = 0;
    end
    
%   inicializa��o do erro m�dio quadr�tico
    EMQ = 0;
    
    %   PEGANDO AMOSTRAS COM PRESEN�A DE RADIA��O PARA O K-MEANS
    k = 1;
    for j = 1:C
        if( d(model(2), j) == 1 )
            xk(:, k) = x(:, j);
            k = k + 1;
        end
    end
    
%   Dimens�es da matriz de amostras
    [Lk, Ck] = size(xk);
    
%   ------------- Inicializa��o Pesos -------------
    w{2, 1} = rand( model(2), model(1) + 1 ); % + 1 devido ao bias de cada neur�nio da camada de sa�da
 
    w{1, 1} = zeros( model(1) , L-1 );
    for nNeuron = 1 : model(1)
%           cada neur�nio tem o centro definido pelas coordenadas na linha
            w{1, 1}( nNeuron , : ) = xk( 2:L, nNeuron);
    end
    w_0 = w;
%   --------------------------
    
%   ************ K-MEANS - TREINAMENTO CAMADA OCULTA ************

%   Executa at� n�o ter diferen�a no grupo de cada amostra
    while( sum( (grupoAtual - grupoAnterior).^2 ) ~= 0 )
        
%       Armazena os grupos de cada amostra da �poca anterior em
%       grupoAnterior para efeitos de controle do c�digo
        grupoAnterior = grupoAtual;
        
%       Calcula a dist�ncia euclidiana entre cada amostra e os pesos       
        for nPeso = 1:model(1)
            for nAmostra = 1:Ck
                for nCoordenada = 1 : Lk-1
                    diferencas(nCoordenada, nAmostra) = ( w{1,1}(nPeso, nCoordenada) - xk(nCoordenada + 1, nAmostra) ).^2;
                end
                somaDiferencas = sum(diferencas);

                % Dist�ncias de cada amostra at� o centro. Cada linha � um
                % neur�nio
                distancias{nPeso,1}(1, nAmostra) = sqrt( somaDiferencas(:,nAmostra) );
            end
        end
        

%       armazena em grupo, de acordo com o n� da amostra,
%       o grupo a qual ela pertence, que � aquele em que ela est�
%       mais pr�xima do peso (centro)
%       depois, armazena os valores desse grupo em grupoAtual apenas
%       para controle do algoritmo
        for nAmostra_2 = 1:Ck
            for nPeso_2 = 1:model(1)
                dists_amostra(1, nPeso_2) = distancias{nPeso_2,1}(1, nAmostra_2); 
            end
            [menorValor, idx] = min(dists_amostra); % pega o menor valor de dist�ncia
            grupos(1, nAmostra_2) = idx;
            grupoAtual = grupos;
        end
        
%       Realiza a atualiza��o do peso (centro)
%       para cada peso, pega as amostras pertencetes a seu grupo
%       soma as coordenadas de cada dimens�o
%       divide a soma de cada coordenada pelo n�mero de amostras dentro do
%       grupo
%       
        for nPeso_3 = 1:model(1)
            for nAmostra_3 = 1:Ck
                if( grupos(1, nAmostra_3) == nPeso_3 ) % se o elemento do grupo for igual ao n�mero do peso:
                    soma{nPeso_3, 1}(:, nAmostra_3) = xk(2:L, nAmostra_3); % armazena em soma todos os elementos pertencentes ao grupo do peso nPeso_3
                    nAmPG = nAmPG + 1; % armazena o n�mero de amostras por grupo
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
    
%   Ap�s definir os centros, calcula a vari�ncia p/ cada neur�nio ap�s
%   fazer a diferen�a entre cada amostra e o centro para cada neur�nio
    for nPeso_4 = 1 : model(1)
        for nAmostra_4 = 1:Ck
              if( grupos(1, nAmostra_4) == nPeso_4 ) % se o elemento do grupo for igual ao n�mero do peso:
                  nElemen{nPeso_4, 1} = nElemen{nPeso_4, 1} + 1; % armazena o n�mero de amostras por grupo
                  difAMC(:, nElemen{nPeso_4,1}) = ( xk(2:L, nAmostra_4) - w{1,1}(nPeso_4, :)' ).^2;
              end
        end
          som = sum(difAMC);
          som2 = sum(som);
          vari(nPeso_4, 1) = 1/nElemen{nPeso_4, 1} * som2
    end

%   Treinamento da camada de sa�da
    while( abs(erroAtual - erroAnterior) >= sigma && epcs <= maxEpcs )
        
        erroAnterior = erroAtual;
        
        
%       APLICA��O DAS ENTRADAS NA CAMADA OCULTA
        for nPeso_5 = 1:model(1)
            for nAmostra_5 = 1:C
%               aplica a fun��o de ativa��o para cada amostra em cada
%               neur�nio.
                gu{nPeso_5,1}(1, nAmostra_5) = exp( (-1) * norm( x(2:L, nAmostra_5) - w{1,1}(nPeso_5, :)' )^2 / ( 2*vari(nPeso_5, 1) ) );
            end
        end
        
%       APLICA��O DAS SA�DAS DA CAMADA OCULTA NA CAMADA DE SA�DA
       for nAmostra_6 = 1:C
           for nPeso_6 = 1:model(1)
               vetorAmostras(nPeso_6, nAmostra_6) = gu{nPeso_6, 1}(1,nAmostra_6);
           end
       end
       
       vetorAmostras_bias = [ (-1)*ones( 1, size(vetorAmostras, 2) ) ; vetorAmostras];
       
%      sa�das da camada de sa�da
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
    title('Gr�fico de Erro','LineWidth', 2)
    legend('�rro M�dio Quadr�tico')
    xlabel(['�pocas [n = ' num2str(epcs) ']'])
    ylabel('EMQ')
    
end