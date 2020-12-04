package komputation

import com.komputation.cpu.network.network
import com.komputation.demos.mnist.MnistData
import com.komputation.initialization.gaussianInitialization
import com.komputation.instructions.continuation.activation.Activation
import com.komputation.instructions.continuation.dense.dense
import com.komputation.instructions.entry.input
import com.komputation.instructions.loss.crossEntropyLoss
import com.komputation.optimization.historical.momentum
import java.io.File
import java.util.*

// loads 55 000 observations due to github limitation with 100 mb max file size
const val TRAIN_PATH = "src/main/resources/datasets/train/mnist_train.csv"

const val TEST_PATH = "src/main/resources/datasets/test/mnist_test.csv"

/**
 * Copied from https://github.com/sekwiatkowski/komputation/blob/master/src/main/kotlin/com/komputation/cpu/demos/mnist/MnistMinimal.kt
 */
fun main() {
    val random = Random(1)

    val numberIterations = 30
    val batchSize = 64

    val (trainingInputs, trainingTargets) = MnistData.loadMnistTraining(File(TRAIN_PATH), true)
    val (testInputs, testTargets) = MnistData.loadMnistTest(File(TEST_PATH), true)

    val inputDimension = 784
    val numberCategories = MnistData.numberCategories

    val initialization = gaussianInitialization(random, 0.0f, 0.1f)
    val optimizer = momentum(0.005f, 0.1f)

    val network = network(
        batchSize,
        input(inputDimension),
        dense(numberCategories, Activation.Softmax, initialization, optimizer)
    )

    val test = network
        .test(
            testInputs,
            testTargets,
            batchSize,
            numberCategories
        )

    network.training(
        trainingInputs,
        trainingTargets,
        numberIterations,
        crossEntropyLoss()
    ) { _: Int, _: Float ->
        println(test.run())
    }
        .run()
}
