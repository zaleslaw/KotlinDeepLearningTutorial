package tensorflow.training.linreg

import org.tensorflow.Graph
import org.tensorflow.Session
import org.tensorflow.Shape
import org.tensorflow.Tensor
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Placeholder
import org.tensorflow.op.core.Variable
import tensorflow.inference.printTFGraph
import kotlin.random.Random

/** Amount of data points. */
private const val n = 10

/** This value is used to fill the Y placeholder in prediction. */
private const val NO_MEANING_VALUE_TO_PUT_IN_PLACEHOLDER = 2000f

fun main() {
    // Prepare the X data
    val xValues = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f)
    // Prepare the Y data. The Y data is created approximately by the next law: 10*x + 2 + noise.
    val yValues = floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f)

    for ((i, x) in xValues.withIndex()) {
        yValues[i] = 10 * x + 2 + Random(42).nextDouble(-0.1, 0.1).toFloat()
    }

    Graph().use { graph ->
        val tf = Ops.create(graph)

        // Define placeholders
        val X = tf.placeholder(Float::class.javaObjectType, Placeholder.shape(Shape.scalar()))
        val Y = tf.placeholder(Float::class.javaObjectType, Placeholder.shape(Shape.scalar()))

        // Define variables
        val weight: Variable<Float> = tf.variable(Shape.scalar(), Float::class.javaObjectType)
        val bias: Variable<Float> = tf.variable(Shape.scalar(), Float::class.javaObjectType)

        // Init variables
        val weightInit = tf.assign(weight, tf.constant(1f))
        val biasInit = tf.assign(bias, tf.constant(1f))

        // Define the model function weight*x + bias
        val mul = tf.math.mul(X, weight)
        val yPredicted = tf.math.add(mul, bias)

        // Define MSE
        val sum = tf.math.pow(tf.math.sub(yPredicted, Y), tf.constant(2f))
        val mse = tf.math.div(sum, tf.constant(2f * n))

        val gradients = tf.gradients(mse, listOf(weight, bias))

        val alpha = tf.constant(0.2f)

        val weightGradientDescent =
            tf.train.applyGradientDescent(weight, alpha, gradients.dy<Float>(0))
        val biasGradientDescent =
            tf.train.applyGradientDescent(bias, alpha, gradients.dy<Float>(1))

        printTFGraph(graph)

        Session(graph).use { session ->
            // Initialize graph variables
            session.runner()
                .addTarget(weightInit)
                .addTarget(biasInit)
                .run()

            // Train the model on data
            for ((cnt, x) in xValues.withIndex()) {
                val y = yValues[cnt]
                Tensor.create(x).use { xTensor ->
                    Tensor.create(y).use { yTensor ->
                        session.runner()
                            .addTarget(weightGradientDescent)
                            .addTarget(biasGradientDescent)
                            .feed(X.asOutput(), xTensor)
                            .feed(Y.asOutput(), yTensor)
                            .run()
                        println("$x $y")
                    }
                }
            }

            // Extract the weight value
            val weightValue = session.runner()
                .fetch("Variable")
                .run()[0].floatValue()

            println("Weight is $weightValue")

            // Extract the bias value
            val biasValue = session.runner()
                .fetch("Variable_1")
                .run()[0].floatValue()

            println("Bias is $biasValue")

            // Let's predict y for x = 10f
            val x = 10f
            var predictedY = 0f
            Tensor.create(x).use { xTensor ->
                Tensor.create(NO_MEANING_VALUE_TO_PUT_IN_PLACEHOLDER).use { yTensor ->
                    predictedY = session.runner()
                        .feed(X.asOutput(), xTensor)
                        .feed(Y.asOutput(), yTensor)
                        .fetch(yPredicted)
                        .run()[0].floatValue()
                }
            }

            println("Predicted value: $predictedY")
        }
    }
}