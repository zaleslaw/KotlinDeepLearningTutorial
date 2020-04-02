package tensorflow.old_api.training.linreg

import org.tensorflow.Graph
import org.tensorflow.Session
import org.tensorflow.Shape
import org.tensorflow.Tensor
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Assign
import org.tensorflow.op.core.Placeholder
import org.tensorflow.op.core.Variable
import org.tensorflow.op.math.Add
import org.tensorflow.op.math.Div
import org.tensorflow.op.train.ApplyGradientDescent
import tensorflow.old_api.inference.printTFGraph
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

        val (X, Y) = placeholders(tf)

        val (weight: Variable<Float>, bias: Variable<Float>) = variables(
            tf
        )

        val (weightInit, biasInit) = initVariables(
            tf,
            weight,
            bias
        )

        val yPredicted = defineModelFunction(tf, X, weight, bias)

        val mse = lossFunction(tf, yPredicted, Y)

        val (weightGradientDescent, biasGradientDescent) = gradients(
            tf,
            mse,
            weight,
            bias
        )

        printTFGraph(graph)

        Session(graph).use { session ->
            train(
                session,
                weightInit,
                biasInit,
                weightGradientDescent,
                biasGradientDescent,
                X,
                Y,
                xValues,
                yValues
            )

            printLinearModel(session)

            // Let's predict y for x = 10f
            val x = 10f
            val predictedValue =
                predict(x, session, X, Y, yPredicted)

            println("Predicted value: $predictedValue")
        }
    }
}

private fun gradients(
    tf: Ops,
    mse: Div<Float>?,
    weight: Variable<Float>,
    bias: Variable<Float>
): Pair<ApplyGradientDescent<Float>, ApplyGradientDescent<Float>> {

    val gradients = tf.gradients(mse, listOf(weight, bias))

    val alpha = tf.constant(0.2f)

    val weightGradientDescent =
        tf.train.applyGradientDescent(weight, alpha, gradients.dy<Float>(0))
    val biasGradientDescent =
        tf.train.applyGradientDescent(bias, alpha, gradients.dy<Float>(1))

    return Pair(weightGradientDescent, biasGradientDescent)
}

private fun defineModelFunction(
    tf: Ops,
    X: Placeholder<Float>,
    weight: Variable<Float>,
    bias: Variable<Float>
): Add<Float>? {
    val mul = tf.math.mul(X, weight)
    val yPred = tf.math.add(mul, bias)
    return yPred
}

private fun lossFunction(
    tf: Ops,
    yPred: Add<Float>?,
    Y: Placeholder<Float>
): Div<Float>? {
    val sum = tf.math.pow(tf.math.sub(yPred, Y), tf.constant(2f))
    val mse = tf.math.div(sum, tf.constant(2f * n))
    return mse
}

private fun predict(
    x: Float,
    session: Session,
    X: Placeholder<Float>,
    Y: Placeholder<Float>,
    yPred: Add<Float>?
): Float {
    var predictedY = 0f
    Tensor.create(x).use { xTensor ->
        Tensor.create(NO_MEANING_VALUE_TO_PUT_IN_PLACEHOLDER).use { yTensor ->
            predictedY = session.runner()
                .feed(X.asOutput(), xTensor)
                .feed(Y.asOutput(), yTensor)
                .fetch(yPred)
                .run()[0].floatValue()
        }
    }
    return predictedY
}

private fun printLinearModel(session: Session) {
    val weightValue = session.runner()
        .fetch("Variable")
        .run()[0].floatValue()

    println("Weight is $weightValue")

    val biasValue = session.runner()
        .fetch("Variable_1")
        .run()[0].floatValue()

    println("Bias is $biasValue")
}

private fun train(
    session: Session,
    weightInit: Assign<Float>,
    biasInit: Assign<Float>,
    weightGradientDescent: ApplyGradientDescent<Float>?,
    biasGradientDescent: ApplyGradientDescent<Float>?,
    X: Placeholder<Float>,
    Y: Placeholder<Float>,
    xValues: FloatArray,
    yValues: FloatArray
) {
    // Initialize graph variables
    session.runner()
        .addTarget(weightInit)
        .addTarget(biasInit)
        .run()

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
}

private fun initVariables(
    tf: Ops,
    weight: Variable<Float>,
    bias: Variable<Float>
): Pair<Assign<Float>, Assign<Float>> {
    val weightInit = tf.assign(weight, tf.constant(1f))
    val biasInit = tf.assign(bias, tf.constant(1f))
    return Pair(weightInit, biasInit)
}

private fun variables(tf: Ops): Pair<Variable<Float>, Variable<Float>> {
    val weights: Variable<Float> =
        tf.variable(Shape.scalar(), Float::class.javaObjectType)
    val bias: Variable<Float> =
        tf.variable(Shape.scalar(), Float::class.javaObjectType)
    return Pair(weights, bias)
}

private fun placeholders(tf: Ops): Pair<Placeholder<Float>, Placeholder<Float>> {
    val X = tf.placeholder(Float::class.javaObjectType, Placeholder.shape(Shape.scalar()))
    val Y = tf.placeholder(Float::class.javaObjectType, Placeholder.shape(Shape.scalar()))
    return Pair(X, Y)
}

