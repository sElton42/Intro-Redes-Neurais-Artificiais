Contém os códigos desenvolvidos em Matlab para alguns tipos de Redes Neurais Artificiais, no contexto da disciplina de Introdução às Redes Neurais Artificiais.


<h1> PERCEPTRON CAMADA SIMPLES </h1>

trainPercep.m: Função de treino de uma rede Perceptron de camada única e neurônio único.

EP1.m: Aplicação do Perceptron Camada Simples na resolução de um problema.

<h1 style="color:blue;"> PERCEPTRON MULTICAMADAS </h1>

BackPropOnPMC.m: Função de treino de uma rede Perceptron Multicamadas com quantidade de neurônios e camadas definido pelo usuário. Utiliza a função logística no treinamento.

BackPropOnPMCTANH.m: Função de treino de uma rede Perceptron Multicamadas com quantidade de neurônios e camadas definido pelo usuário. Utiliza a função tangente hiperbólica no treinamento.

dataImport.m: Função para importar dados de um arquivo CSV para poder usar com o Perceptron Multicamadas.

diabetes.csv: Arquivo CSV contendo dados de pessoas diabéticas, obtido em uma base de dados pública, para servir de testes do treinamento da rede neural.

OperationLOG.m: Script para testar o Perceptron Multicamadas usando função logística, com os dados do arquivos diabetes.csv.

OperationTANH.m: Script para testar o Perceptron Multicamadas usando função tangente hiperbólica com os dados do arquivos diabetes.csv.
