function [W1_atual W2_atual epoca EM_Atual E] = pmc_TDNN_M(DADOS,n,ee,np,alf,NEU)
% ELTON SOARES SILVA - 41711ETE010
% SAMUEL D. LIMA     - 41621ETE011
%% MANIPULAÇÃO DOS DADOS DE ENTRADA DE ACORDO COM O NP

X=[];

for k=1:length(DADOS)-np
    X(k,:) = [-1 DADOS(k:k+np-1,1)'];
end

d = DADOS(np+1:length(DADOS))';

% INICIALIZAÇÃO DE VALORES PARA MATRIZ DE PESOS 
    dimx = size(X);                %Armazena as dimensões da Matriz De dados de Entrada
    W1_atual = rand(NEU,(np+1));   % -> "NEU" neurônios e np+1 entradas (bias)
    W2_atual = rand(1,NEU+1);      % -> "NEU" mais o bias para o neurônio da camada de saída
    W1_pas = W1_atual;
    W2_pas = W2_atual;
    
    dimx = size(X);    %Armazena as dimensões da Matriz De dados de Entrada
    dim1 = size(W1_atual);
    dim2 = size(W2_atual);
    
    EM_Anterior=0;     % Armazena erro da época anterior
    cond=1;            % Var auxiliar
    epoca=0;           % Inicia contador de épocas
%------------------------------------------------------------------------------------------------------% 

%% TREINAMENTO
while cond == 1
    
    %% PROPAGATION
    
    for a=1:dimx(1) % Percorre todas as Linhas (Amostras p)
        
    % -> 1° Camada Escondida
        I1 = X(a,:)*W1_atual';  % somatório: entradas * pesos + bias
        Y1 = (1./(1+exp(-I1))); % Função de ativação
        Y1 = [-1,Y1];           % concatena multiplicador do bias p/ próxima camada
        
    % -> 2° Camada (camada de saída)
        I2 = Y1*W2_atual';          % somatório: Y1 * pesos + bias
        Y2(a) = (1./(1+exp(-I2)));  % Função de ativação
        
        E(a) = 0.5*sum((d(1,a)-Y2(a)).^2,'all'); %Cálculo do erro para cada Amostra 

        %% Backward - BACKPROPAGATION
        
        D2 = (d(1,a)-Y2(a))*((exp(-I2)/((1+exp(-I2))^2))); % Cálculo do Delta
 
        % Ajuste dos pesos entre camada escondida e camada de saída com
        % momentum
        
        for a3=1:dim2(2)
            W2_fut(:,a3) = W2_atual(:,a3)+(alf*(W2_atual(:,a3)-W2_pas(:,a3)))+n.*(D2').*(Y1(a3));
        end
        
        W2_pas = W2_atual;
        W2_atual = W2_fut;
        
        % Ajuste dos pesos entre camada de entrada e camada escondida
        
        for bb=1:dim1(1)
           D1(bb) = -sum(D2'.*W2_atual(:,bb),'all')*(exp(-I1(bb))/((1+exp(-I1(bb)))^2));
        end
        for b=1:dim1(1)
            W1_fut(b,:) = ((W1_atual(b,:))+(alf*(W1_atual(b,:)-W1_pas(b,:))) + n.*D1(b).*X(a,:));
        end
        
        W1_pas = W1_atual;
        W1_atual = W1_fut;
        
    end
    %% Teste para saber se para o algoritmo:
    
    epoca = epoca + 1;
    
    EM_Atual(epoca) = (1/dimx(1))*sum(E,'all'); % Erro Médio Quadrático
    
    Err(epoca) = abs(EM_Atual(epoca)-EM_Anterior);

    if Err(epoca)<ee
        cond=0;
    else
        EM_Anterior = EM_Atual(epoca);          %Realoca os Resultados
    end
end

%% SAÍDAS

figure
plot(Y2,'r')
hold on
plot(d,'b')
title([' Treinamento np=' num2str(np) + " e N=" num2str(NEU)],'FontSize',12,'FontWeight','bold','Color','black')
xlabel('t[n]','FontSize',12,'FontWeight','bold','Color','black')
ylabel('Resultado y(t)','FontSize',12,'FontWeight','bold','Color','black')
legend('y(t)','x(t)')

return
end