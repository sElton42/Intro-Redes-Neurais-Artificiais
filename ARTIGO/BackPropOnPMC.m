function [w, w_0, epcs, EMQ, accuracyTrain] = BackPropOnPMC(x, d, eta, sigma, maxEpcs, model)

    [L, C] = size(x);             % dimensões da matriz de amostras
    epcs          = 0;            % inicializando contador de épocas
    erroAtual     = 123456789;    % inicialização de um valor qualquer grande para o erro quadrático médio atual
    erroAnterior  = 0;            % inicialização da variável que receberá o erro quadrático médio da época anterior
   
    % **************** Inicializa todos os pesos dos neurônios da 1° Camada
    w{1,1} = rand(model(1), L); 
    % Termina de inicializar os pesos dos neurônios das camadas posteriores
    for i = 2 : numel(model)
        w{i, 1} = rand( model(i), model(i-1) + 1 ); % + 1 se deve ao limiar
    end
    % Guarda os valores de pesos e limiar gerados aleatoriamente
    w_0 = w;           
    %  ****************************************
    
    %% ------------------------- TREINAMENTO PROPRIAMENTE DITO
    
    while ( (abs(erroAtual - erroAnterior) >= sigma) && epcs <= maxEpcs )
        
        erroAnterior = erroAtual;
        
        for numAmostra = 1:C
            % PROPAGATION
            amostra = x(:,numAmostra);
            amostra = amostra';
            saidaDesejada = d(:, numAmostra);
            
            for numLayer = 1:numel(model)
                w_ = w{numLayer, 1};
                w_ = w_';
                
                if(numLayer == 1)
                    u_ = amostra * w_;
                    u{1,1}(:,numAmostra) = u_';
                    y_ = 1 ./ ( 1 + exp( -u_ ) );
                    y{1,1}(:,numAmostra) = y_';  
                    y_ = [-1 , y_ ];
                else
                    u_ = y_ * w_;
                    u{numLayer,1}(:,numAmostra) = u_';
                    y_ = 1 ./ ( 1 + exp( -u_ ) );
                    y{numLayer,1}(:,numAmostra) = y_';
                    y_ = [-1 , y_ ];
                end
                
                % BACKPROPAGATION
                if( numLayer == numel(model) )
                    for numLayerBack = numLayer : -1 : 1
                        w_ = w{numLayerBack, 1};
                        w_ = w_';
                        if(numLayerBack == numel(model))
                            erro   =  saidaDesejada - y{numLayerBack, 1}(:,numAmostra);
                            gLinha = ( exp( -u{numLayerBack, 1}(:,numAmostra) ) ...
                                ./ ((1+exp( -u{numLayerBack, 1}(:, numAmostra) )).^2) );
                            delta_ =  erro .* gLinha;
                            delta{numLayerBack, 1}(:, numAmostra) = delta_;
                            yAux = [-1 ; y{numLayerBack-1,1}(:, numAmostra)];
                            w_ = w_';
                            for i = 1 : size(w_, 2)
                                w_(:,i) = w_(:,i) + eta .* delta_ .* yAux(i);
                            end
                            w_ = w_';
                            w{numLayerBack, 1} = w_';
                        else
                            if(numLayerBack == 1)
                                clear delta_
                                for j = 1:size(w{numLayerBack, 1},1)
                                    gLinha = ( exp( -u{numLayerBack, 1}(j,numAmostra) ) ...
                                    ./ ((1+exp( -u{numLayerBack, 1}(j, numAmostra) )).^2) );
                                
                                    delta_(j) = - sum(delta{numLayerBack+1,1}(:, numAmostra) .* w{numLayerBack+1,1}(:,j)) * gLinha;
                                end
                                delta{numLayerBack, 1}(:, numAmostra) = delta_;
                                w_ = w_';
                                for i = 1 : size(w_, 1)
                                    w_(i, :) = w_(i, :) + eta .* delta_(1, i) .* amostra;
                                end
                                w_ = w_';
                                w{numLayerBack, 1} = w_';
                            else
                                clear delta_
                                for j = 1:size(w{numLayerBack, 1},1)
                                    gLinha = ( exp( -u{numLayerBack, 1}(j,numAmostra) ) ...
                                    ./ ((1+exp( -u{numLayerBack, 1}(j, numAmostra) )).^2) );
                                
                                    delta_(j) = - sum(delta{numLayerBack+1,1}(:, numAmostra) .* w{numLayerBack+1,1}(:,j)) * gLinha;
                                end
                                delta{numLayerBack, 1}(:, numAmostra) = delta_;
                                w_ = w_';
                                for i = 1 : size(w_, 1)
                                    w_(i,:) = w_(i,:) + eta .* delta_(1, i) .* y{numLayerBack,1}(i, numAmostra);
                                end
                                w_ = w_';
                                w{numLayerBack, 1} = w_';
                            end                                         
                        end
                    end
                end
            end
        end
        epcs = epcs + 1
        erroPorAmostra = sum(0.5 * (d - y{numel(model), 1}).^2);
        EMQ(epcs) = 1/C * sum(erroPorAmostra);
        erroAtual = EMQ(epcs);
    end
    
    %% CHECAGEM DE ACERTOS DO TREINAMENTO (1 neurônio)
%     accuracyTrain = 0;
%     for nAmos = 1:C
%         if( round(y{numel(model),1}(1,nAmos)) == d(1,nAmos) ), accuracyTrain = accuracyTrain + 1; end
%     end
%     accuracyTrain
    
    %% CHECAGEM DE ACERTOS DO TREINAMENTO (2 neurônios)
    accuracyTrain = 0;
    for nAmos = 1 : size(d,2)
        for nLay = 1 : 2
            saida(nLay, nAmos) = round(y{numel(model),1}(nLay,nAmos));
        end
    end

    for nAmos = 1 : size(d,2)
        if( ( saida(1,nAmos) == d(1,nAmos) ) & ( saida(2,nAmos) == d(2,nAmos) ) )
            accuracyTrain = accuracyTrain + 1;
        end
    end
    accuracyTrain
end