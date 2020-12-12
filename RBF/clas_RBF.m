function [resp] = clas_RBF(w, sigma, x)
% Definição
[l,c] = size(x); % Dimensões da matriz de entrada (IN)

% Matriz de peso 1
W1 = w{1,1}';%[X(1,:);X(2,:)];
p1 = size(W1);

% Matriz de peso 2
W2= w{2,1};
p2 = size(W2);

    for i=1:l % Varredura de P amostras
        % 1° Camada
        for j=1:p1(1)
            Y1(j) = exp(-sum((x(i,:)-W1(j,:)).^2,"all")/(2*(sigma(j)^2))); %Resultado de cada k neurônio
        end
        Y2(i) = [-1 Y1]*W2'; %Saída da Rede RBF (OUT)
    end
    resp = Y2';

    for k1=1:10 %Pós-Processamento
        if Y2(k1) >= 0
            resp(k1,2) = 1;
        else
            resp(k1,2) = -1
        end
    end
end