Contém os códigos desenvolvidos em Matlab para alguns tipos de Redes Neurais Artificiais, no contexto da disciplina de Introdução às Redes Neurais Artificiais.


<h1> PERCEPTRON CAMADA SIMPLES </h1>

*trainPercep.m:* Função de treino de uma rede Perceptron de camada única e neurônio único.

*EP1.m:* Aplicação do Perceptron Camada Simples na resolução de um problema.

<h1> PERCEPTRON MULTICAMADAS </h1>

*BackPropOnPMC.m:* Função de treino de uma rede Perceptron Multicamadas com quantidade de neurônios e camadas definido pelo usuário. Utiliza a função logística no treinamento.

*BackPropOnPMCTANH.m:* Função de treino de uma rede Perceptron Multicamadas com quantidade de neurônios e camadas definido pelo usuário. Utiliza a função tangente hiperbólica no treinamento.

*dataImport.m:* Função para importar dados de um arquivo CSV para poder usar com o Perceptron Multicamadas.

*diabetes.csv:* Arquivo CSV contendo dados de pessoas diabéticas, obtido em uma base de dados pública, para servir de testes do treinamento da rede neural.

*OperationLOG.m:* Script para testar o Perceptron Multicamadas usando função logística, com os dados do arquivo diabetes.csv.

*OperationTANH.m:* Script para testar o Perceptron Multicamadas usando função tangente hiperbólica com os dados do arquivos diabetes.csv.

<h1> REDE RBF (Radial Basis Function Network) </h1>

*trainRBF.m:* Função de treino de uma rede de Função de Base Radial (RBF).

*OperationRBF.m:* Script para testar a rede de Função de Base Radial (RBF) com os dados do arquivo diabetes.csv.

*dataImport.m:* Função para importar dados de um arquivo CSV para poder usar com a rede RBF.

*diabetes.csv:* Arquivo CSV contendo dados de pessoas diabéticas, obtido em uma base de dados pública, para servir de testes do treinamento da rede neural.

<h1> ADALINE (Adaptive Linear Element) </h1>

*trainADALINE.m:* Função de treino de uma rede ADALINE.

*EP2.m:* Script para testar a rede ADALINE.

*dataTrainIMP:* Função para importar os dados de treino da rede ADALINE.

*dadosTreinamento.csv:* Dados de treinamento.

*dataNewIMP:* Função para importar os dados de teste da rede ADALINE treinada.

*dadosNovos.csv:* Dados de teste da rede treinada.

