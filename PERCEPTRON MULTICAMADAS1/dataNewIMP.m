function x = dataNewIMP(nomearq, numRows, numCols)
% Importa dados novos de arquivos CSV, retornando em:
% x:    entradas
% 
% Recebendo:
% nomearq:  nome do arquivo
% numRows:  número de linhas contendo dados
% numCols:  número de colunas contendo dados

% ELTON SOARES SILVA - 41711ETE010.
% SAMUEL D. LIMA     - 41621ETE011

% abre o arquivo de dados no modo leitura e pula a primeira linha
% já que ela contém o header (cabeçalho)
file = fopen(nomearq ,'r');
fgetl(file);

% inicializa a matriz de dados e o contador de linhas kLine
data = zeros(numRows, numCols);
kLine = 1;

% a cada iteração, pega uma linha do arquivo, checa se é o fim ("end")
% se for, para o laço while
% se não for, divide a linha em células, uma p/ cada dado de entrada
% dentro do for, converte os dados dentro de cada célula p/ valores
% numéricos (double) e depois armazena em data, na posição correta, 
% dada pelo n° da linha kLine e n° da coluna kCol.
% depois, incrementa kLine, passando para a próxima linha
while 1
    fline  = fgetl(file);
    
    if (strcmpi(fline(1:3), 'end')), break; end
    
    aline = regexp( fline, ';','split' );

    for kCol = 1:numCols
        data(kLine, kCol) = str2double(aline{1,kCol});
    end

    kLine = kLine + 1;
end

% armazena em x, os dados de entrada de cada amostra
% depois, faz a transposta
x = data(:, 1 : numCols);
x = x';

% Concatena matriz linha de uns com a matriz de dados x
[L, C] = size(x);
uns = (-1) * ones(1, C);
x = [uns ; x];
x = x';

% fecha arquivo de dados aberto
fclose(file);