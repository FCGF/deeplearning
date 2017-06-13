package br.org.catolicasc.deeplearning;

import java.io.IOException;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class Mnist {
	
	public static void main(String[] args) throws IOException {
		
		//No exemplo do MNIST cada imagem possui 28 x 28 pixeis
        final int qtdeLinhas = 28;
        final int qtdeColunas = 28;
        int qtdeSaida = 10; // quantidade de classes de saida, numerais 0 a 9
        
        //Obtidos por experimentação
        int tamanhoLote = 128; // quantidade de processamento para cada passo, maior número = mais rápido.
        int qtdeEpocas = 15; // quantidades de epocas a serem executadas, + épocas = maior acuracia e mais lento

        int seed = 123; // número não aleatório para que seja reproduzivel
         
        //Obtem os DataSetIterators, utilizados para obter a informação do conjunto de datos do MNIST.
        
        //Utilizado para treinar o modelo
        DataSetIterator mnistTrain = new MnistDataSetIterator(tamanhoLote, true, seed);
        
        //Utilizado para avaliar o modelo gerado
        DataSetIterator mnistTest = new MnistDataSetIterator(tamanhoLote, false, seed);

        //Quanto mais camadas, maior a capacidade da rede de perceber complexidade e detalhes e melhora acuracia
        System.out.println("Construindo modelo");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed) //usado para gerar pesos iniciais aleatórios
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) //algoritmo de optimização de custo e diminuição de erros
                .iterations(1) //Passos de aprendizado, quanto mais iterações, maior aprendizado e menor erro
                .learningRate(0.006) //Modificação feita nos pesos por cada iteração. Quanto menor, menor o erro, mas mais iterações necessárias
                .updater(Updater.NESTEROVS).momentum(0.9) //afeta em que direção os pesos são ajustados
                .regularization(true).l2(1e-4) //usado para prevenir overfitting, tira importancia de pesos individuais
                .list() //usado para criar as camadas
                .layer(0, new DenseLayer.Builder() //camada de entrada
                        .nIn(qtdeLinhas * qtdeColunas)
                        .nOut(1000)
                        .activation(Activation.RELU)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) //camada oculta
                        .nIn(1000)
                        .nOut(qtdeSaida)
                        .activation(Activation.SOFTMAX)
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .pretrain(false).backprop(true) //usar backpropagation para ajustar pesos
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        //print the score with every 1 iteration
        model.setListeners(new ScoreIterationListener(1));

        System.out.println("Treinando modelo");
        for( int i=0; i<qtdeEpocas; i++ ){
            model.fit(mnistTrain);
        }


        System.out.println("Avaliando modelo");
        Evaluation eval = new Evaluation(qtdeSaida); //create an evaluation object with 10 possible classes
        while(mnistTest.hasNext()){
            DataSet next = mnistTest.next();
            INDArray output = model.output(next.getFeatureMatrix()); //get the networks prediction
            eval.eval(next.getLabels(), output); //check the prediction against the true class
        }

        System.out.println(eval.stats());
        
	}
	
}
