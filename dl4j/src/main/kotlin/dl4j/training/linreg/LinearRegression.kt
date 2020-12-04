package dl4j.training.linreg

import org.nd4j.autodiff.samediff.SDVariable
import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.linalg.api.buffer.DataType
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.weightinit.impl.XavierInitScheme
import java.util.*

/**
 *  loss = MSE(output, label) = 1/(nExamples * nOut) * sum_i (labels_i - out_i) ^ 2
 *  output = input * weights + bias
 */
fun main() {
    // How to calculate gradients, and get gradient arrays - linear regression (MSE, manually defined)
    val nIn = 4L
    val nOut = 2L

    val sd = SameDiff.create()

    // Step 1: Create our placeholders. Shape: [minibatch, in/out]
    val input = sd.placeHolder("input", DataType.FLOAT, -1, nIn)
    val labels = sd.placeHolder("labels", DataType.FLOAT, -1, 1)

    // Step 2: Create our variables
    val xavierInitScheme = XavierInitScheme('c', nIn.toDouble(), nOut.toDouble())
    val weights: SDVariable = sd.`var`("weights", xavierInitScheme, DataType.FLOAT, nIn, nOut)
    val bias: SDVariable = sd.`var`("bias", 1)

    // Step 3: Define our forward pass
    val out = input.mmul(weights).add(bias)

    // Step 4: Define loss function
    val difference = labels.sub(out)
    val sqDiff = sd.math().square(difference)
    val mse = sqDiff.mean("mse")

    // Step 5: Prepare mock data
    val miniBatchSize = 3
    val seed = 12345L

    val inputShape = intArrayOf(miniBatchSize, nIn.toInt())
    val labelShape = intArrayOf(miniBatchSize, nOut.toInt())

    val inputArr = Nd4j.rand(inputShape, seed)
    val labelArr = Nd4j.rand(labelShape, seed)

    val placeholderData: MutableMap<String, INDArray> = HashMap()
    placeholderData["input"] = inputArr
    placeholderData["labels"] = labelArr

    // Step 6: Execute forward pass to calculate loss function value
    val loss = sd.output(placeholderData, "mse")["mse"]
    println("MSE: $loss")

    // Step 7: Calculate and print out gradients
    val gradMap = sd.calculateGradients(placeholderData, "weights", "bias")

    println("Weights gradient:")
    println(gradMap["weights"])

    println("Bias gradient:")
    println(gradMap["bias"])
}