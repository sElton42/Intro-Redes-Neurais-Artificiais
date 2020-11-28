%% Classifica��o PMC

% Nomes: Samuel Davi de Lima e Elton Soares Silva
% N� Mat: 41621ETE011 | 41711ETE010

% recebe os dados novos para classifica��o e os classifica, usando
% as matrizes de pesos encontradas durante o treinamento

function [y, yNoPos] = clas_PMC(X,W1,W2)
% Armazena as dimens�es da Matriz de dados de Entrada e das matrizes de
% pesos

dimx = size(X);  
dim1 = size(W1);
dim2 = size(W2);

yNoPos = zeros(dimx(1), 3);

% concatena o multiplicador -1 do bias (se necess�rio)
% X = [-ones(dimx(1),1) X];

% Propaga��o das entradas novas a serem classificadas
for a=1:dimx(1) % Percorre todas as Linhas (Amostras p)
    % -> 1� Camada
        I1 = X(a,:)*W1';
        Y1 = (1./(1+exp(-I1)));
        Y1 = [-1,Y1];
    % -> 2� Camada
        I2 = Y1*W2';
        Y2 = (1./(1+exp(-I2)));
        
    % -> Sa�da da Rede
    % Faz o p�s-processamento
        for aa=1:dim2(1)
            if Y2(aa)>=0.5
                y(a,aa)=1;
                yNoPos(a,aa) = Y2(aa);
            else
                y(a,aa)=0;
                yNoPos(a,aa) = Y2(aa);
            end
        end
end
return
end