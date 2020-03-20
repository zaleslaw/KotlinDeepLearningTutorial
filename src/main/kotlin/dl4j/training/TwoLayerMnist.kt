package dl4j.training

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Nesterovs
import org.nd4j.linalg.lossfunctions.LossFunctions

fun main() {
    val numRows = 28
    val numColumns = 28
    val outputNum = 10 // number of output classes
    val batchSize = 64 // batch size for each epoch
    val rngSeed = 123 // random number seed for reproducibility
    val numEpochs = 15 // number of epochs to perform
    val rate = 0.0015 // learning rate

    // Get the DataSetIterators:
    val mnistTrain = MnistDataSetIterator(batchSize, true, rngSeed)
    val mnistTest = MnistDataSetIterator(batchSize, false, rngSeed)

    val conf = NeuralNetConfiguration.Builder()
        .seed(rngSeed.toLong()) //include a random seed for reproducibility
        // use stochastic gradient descent as an optimization algorithm
        .activation(Activation.RELU)
        .weightInit(WeightInit.XAVIER)
        .updater(Nesterovs(rate, 0.98)) //specify the rate of change of the learning rate.
        .l2(rate * 0.005) // regularize learning model
        .list()
        .layer(
            DenseLayer.Builder() //create the first input layer.
                .nIn(numRows * numColumns)
                .nOut(500)
                .build()
        )
        .layer(
            DenseLayer.Builder() //create the second input layer
                .nIn(500)
                .nOut(100)
                .build()
        )
        .layer(
            OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD) //create hidden layer
                .activation(Activation.SOFTMAX)
                .nIn(100)
                .nOut(outputNum)
                .build()
        )
        .build()

    val model = MultiLayerNetwork(conf)
    model.init()

    model.setListeners(
        ScoreIterationListener(5),
        org.deeplearning4j.optimize.listeners.EvaluativeListener(mnistTest, 300)
    )  //print the score every 5 iterations and evaluate periodically
    model.fit(mnistTrain, numEpochs)

    val eval: org.nd4j.evaluation.classification.Evaluation = model.evaluate(mnistTest)
    println(eval.stats())
}