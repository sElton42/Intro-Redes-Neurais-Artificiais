function [x, d] = dataTrainIMP(nomearq, numRows, numCols, numSaidas)
% Importa dados de treinamento de arquivos CSV, retornando em:
% x:    entradas
% d:    sa�das desejadas
% 
% Recebendo:
% nomearq:  nome do arquivo
% numRows:  n�mero de linhas contendo dados
% numCols:  n�mero de colunas contendo dados

% ELTON SOARES SILVA - 41711ETE010.

% abre o arquivo de dados no modo leitura e pula a primeira linha
% j� que ela cont�m o header (cabe�alho)
file = fopen(nomearq ,'r');
fgetl(file);

% inicializa a matriz de dados e o contador de linhas kLine
data = zeros(numRows, numCols);
kLine = 1;

% a cada itera��o, pega uma linha do arquivo, checa se � o fim ("end")
% se for, para o la�o while
% se n�o for, divide a linha em c�lulas, uma p/ cada dado de entrada
% e uma c�lula para a sa�da de cada amostra.
% dentro do for, converte os dados dentro de cada c�lula p/ valores
% num�ricos (double) e depois armazena em data, na posi��o correta, 
% dada pelo n� da linha kLine e n� da coluna kCol.
% depois, incrementa kLine, passando para a pr�xima linha
while 1
    fline  = fgetl(file);
    
    if (strcmpi(fline(1:3), 'end')), break; end
    
    aline = regexp( fline, ',','split' );

    for kCol = 1:numCols
        data(kLine, kCol) = str2double(aline{1,kCol});
    end

    kLine = kLine + 1;
end

% armazena em x, os dados de entrada de cada amostra
% armazena em d, as sa�das desejadas de cada amostra
% depois, faz a transposta
x = data(:, 1 : numCols - numSaidas);
x = x';
d = data( :, (numCols - numSaidas + 1) : numCols );
d = d';

% Concatena matriz linha de uns com a matriz de dados x
[L, C] = size(x);
uns = (-1) * ones(1, C);
x = [uns ; x];

% fecha arquivo de dados aberto
fclose(file);