package tensorflow.old_api.training

import org.tensorflow.*
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Assign
import org.tensorflow.op.core.Placeholder
import org.tensorflow.op.core.Variable
import org.tensorflow.op.math.Mean
import org.tensorflow.op.nn.Softmax


const val learningRate = 0.01
const val trainingEpochs = 1000
const val n = 10 // amount of data points

fun main() {
    Graph().use { graph ->
        val tf = Ops.create(graph)

        val (X, Y) = definePlaceholders(tf)

        val (weights: Variable<Float>, bias: Variable<Float>) = defineModelVariables(tf)

        val (weightsInit, biasesInit) = initModelVariables(tf, weights, bias)

        val mul = tf.math.mul(X, weights)
        val y_pred = tf.math.add(mul, bias)

        //# Mean Squared Error Cost Function
        //cost = tf.reduce_sum(tf.pow(y_pred-Y, 2)) / (2 * n)
        // TODO: https://stackoverflow.com/questions/43130365/valueerror-invalid-reduction-dimension-1-for-input-with-1-dimensions

        val sum = tf.reduceSum(
            tf.math.pow(tf.math.sub(y_pred, Y), tf.constant(2f)), constArray(tf, 1)
        )

        val cost = tf.math.div(sum, tf.constant(2f * n))

        // my own optimizer
        val gradients = tf.gradients(cost, listOf(weights, bias))
        val alpha = tf.constant(0.2f)
        val weightGradientDescent =
            tf.train.applyGradientDescent(weights, alpha, gradients.dy<Float>(0))
        val biasGradientDescent =
            tf.train.applyGradientDescent(bias, alpha, gradients.dy<Float>(1))

        Session(graph).use { session ->
            // Initialize graph variables
            session.runner()
                .addTarget(weightsInit)
                .addTarget(biasesInit)
                .run()

            val xValues = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f)
            val yValues = floatArrayOf(11.1f, 21.2f, 31.3f, 40.7f, 50.8f, 61.1f, 70.95f, 81f, 91.1f, 101.2f)

            for ((cnt, x) in xValues.withIndex()) {

                Tensor.create(x).use { xTensor ->
                    Tensor.create(yValues[cnt]).use { yTensor ->
                        session.runner()
                            .addTarget(weightGradientDescent)
                            .addTarget(biasGradientDescent)
                            .feed(X.asOutput(), xTensor)
                            .feed(Y.asOutput(), yTensor)
                            .run()
                    }
                }
            }


            val predicted: Operand<Long> = tf.math.argMax(cost, tf.constant(1))
            val expected: Operand<Long> = tf.math.argMax(Y, tf.constant(1))

            val accuracy = tf.math.mean(
                tf.dtypes.cast(
                    tf.math.equal<Long>(predicted, expected),
                    Float::class.javaObjectType
                ), constArray(tf, 0)
            )
            println(accuracy)

        }
    }
}



private fun metricGraph(
    tf: Ops,
    cost: Softmax<Float>?,
    labels: Placeholder<Float>
): Mean<Float>? {
    val predicted: Operand<Long> = tf.math.argMax(cost, tf.constant(1))
    val expected: Operand<Long> = tf.math.argMax(labels, tf.constant(1))

    val accuracy = tf.math.mean(
        tf.dtypes.cast(
            tf.math.equal<Long>(predicted, expected),
            Float::class.javaObjectType
        ), constArray(tf, 0)
    )
    return accuracy
}

private fun initModelVariables(
    tf: Ops,
    weights: Variable<Float>,
    bias: Variable<Float>
): Pair<Assign<Float>, Assign<Float>> {
    val weightsInit =
        tf.assign(weights, tf.zeros(constArray(tf, 10), Float::class.javaObjectType))
    val biasInit = tf.assign(bias, tf.constant(1f))
    return Pair(weightsInit, biasInit)
}

private fun defineModelVariables(tf: Ops): Pair<Variable<Float>, Variable<Float>> {
    val weights: Variable<Float> =
        tf.variable(Shape.make(10), Float::class.javaObjectType)
    val bias: Variable<Float> =
        tf.variable(Shape.scalar(), Float::class.javaObjectType)
    return Pair(weights, bias)
}

private fun definePlaceholders(tf: Ops): Pair<Placeholder<Float>, Placeholder<Float>> {
    val X = tf.placeholder(Float::class.javaObjectType, Placeholder.shape(Shape.make(10)))
    val Y = tf.placeholder(Float::class.javaObjectType, Placeholder.shape(Shape.scalar()))
    return Pair(X, Y)
}

