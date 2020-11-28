%% TREINAAMENTO DO PMC
% Nomes: Samuel Davi de Lima e Elton Soares Silva
% N° Mat: 41621ETE011 | 41711ETE010
%------------------------------------------------------------------------------------------------------%
%% PMC
function [W1 W2 epoca Err EM_Atual] = train_PMC(X,d,n,ee)
%OBS:
%X -> Entradas da rede: Deve ser repassado como uma matriz onde cada amostra pk seja na forma do vetor linha. 
%     Ex.: (p1(1,:))
%d -> Saídas Desejadas: Deve ser repassado com que cada amostra pk seja na
%     forma de vetor linha.

%n -> Taxa de aprendizagem
%e -> Precisão Requerida

% RETORNA:
%y          -> Retorna os resultados de Classificação
%W1         -> Matriz de Pesos Finais
%W1 e W2    -> Matrizes a serem repassadas nas Dimensões [N° Neurônios, N° de Entradas]
%epoca      -> número de épocas gastas p/ treinar 
%Err        -> erro atingido durante o treinamento

%------------------------------------------------------------------------------------------------------%

    % Inicializa aleatoriamente a matriz w p/ 1° Camada (entre
    % camada de entrada e 1° camada escondida)
    % OBS.: cada linha tem os pesos e o valor do bias de cada neurônio
    % como são 15 neurônios na 1° camada escondida, temos 15 linhas
    % como temos 4 entradas e 1 bias para cada neurônios, temos 5 colunas
    W1 = rand(15,5);
    
    % Inicializa aleatoriamente a matriz w p/ 2° Camada (entre 
    % 1° camada escondida e a camada de saída)
    % OBS.: cada linha tem os pesos e o valor do bias de cada neurônio
    % como são 3 neurônios na camada de saída, temos 3 linhas
    % como temos 15 entradas e 1 bias para cada neurônio, temos 16 colunas
    W2 = rand(3,16);
    
    % pega as dimensões da matriz de dados de treino e das matrizes de pesos
    dimx = size(X);
    dim1 = size(W1);
    dim2 = size(W2);

    EM_Anterior=0;     % inicializa o erro médio quadrático
    cond=1;            % var auxiliar
    epoca=0;           % inicializa contador de épocas
%------------------------------------------------------------------------------------------------------% 
%% Concatena o multiplicador do bias (-1) com as entradas
% (necessário quando as entradas já não estão concatenadas)
% X = [-ones(dimx(1),1) X];

%% TREINAMENTO
% executa enquanto a precisão não for atingida
while cond == 1
    
    %% Fase Forward
    for a = 1:dimx(1) % Percorre todas as Linhas (Amostras p)
    % -> 1° Camada
    %   Multiplica entradas pelos pesos
        I1 = X(a,:) * W1';
    %   Aplica a função de ativação (Função Logística)
        Y1 = (1./(1+exp(-I1)));
        Y1 = [-1,Y1];
        
    % -> 2° Camada 
    %   multiplica a saída da camada anterior pela matriz de pesos W2
        I2 = Y1*W2';
        
    %   aplica a função de ativação
        Y2 = (1./(1+exp(-I2)));
        
    % -> Saída da Rede c/ pós-processamento
        for aa=1:dim2(1)
            if Y2(aa)>=0.5
                y(a,aa)=1;
            else
                y(a,aa)=0;
            end
        end
        
        % Calcula o erro p/ cada amostra
        E(a) = 0.5*sum((d(a,:)-Y2(:)').^2,'all');

        %% Fase Backward
%       BACKPROPAGATION

        D2 = (d(a,:)-Y2).*((exp(-I2)./((1+exp(-I2)).^2)));
        
        % Ajuste dos pesos entre camada escondida e camada de saída
        for a3=1:dim2(2)
            W2(:,a3) = W2(:,a3)+n.*(D2').*(Y1(a3));
        end
        
        % Ajuste dos pesos entre camada de entrada e camada escondida
        for bb=1:dim1(1)
            D1(bb) = -sum(D2'.*W2(:,bb),'all')*(exp(-I1(bb))/((1+exp(-I1(bb)))^2));
        end
        for b=1:dim1(1)
            W1(b,:) = ((W1(b,:)) + n.*D1(b).*X(a,:));
        end
    end
    
    %% Checagem do Critério de parada
    
    %   incrementa o n° de épocas
    epoca = epoca+1;
    
    % armazena o erro atual da rede durante o treinamento
    EM_Atual(epoca) = (1/dimx(1))*sum(E,'all');
    
    % armazena a diferença entre o erro atual e o da época anterior
    Err(epoca) = abs(EM_Atual(epoca)-EM_Anterior);

    % checa se a precisão foi atingida
    if Err(epoca)<ee
        cond=0;
    else
        EM_Anterior = EM_Atual(epoca); %Realoca os Resultados
    end
end
end