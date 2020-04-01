package tensorflow.old_api.training

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

const val n = 10 // amount of data points
const val NO_MEANING_VALUE_TO_PUT_IN_PLACEHOLDER = 2000f

// 10 * xValues + 1 +- 0.2 = yValues
val xValues = floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f)
val yValues = floatArrayOf(10.9f, 21.2f, 30.8f, 41.2f, 50.8f, 61.1f, 70.95f, 81f, 90.9f, 101.2f)

fun main() {
    Graph().use { graph ->
        val tf = Ops.create(graph)

        val (X, Y) = definePlaceholders(tf)

        val (weight: Variable<Float>, bias: Variable<Float>) = defineModelVariables(tf)

        val (weightInit, biasInit) = initModelVariables(tf, weight, bias)

        val yPred = model(tf, X, weight, bias)

        val mse = lossFunction(tf, yPred, Y)

        val (weightGradientDescent, biasGradientDescent) = gradients(tf, mse, weight, bias)

        printTFGraph(graph)

        Session(graph).use { session ->

            train(session, weightInit, biasInit, weightGradientDescent, biasGradientDescent, X, Y)

            printLinearModel(session)

            val x = 10f
            val res = predict(x, session, X, Y, yPred)

            println("Predicted value: $res")
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

private fun model(
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
    Y: Placeholder<Float>
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
                println(" $x $y")
            }
        }
    }
}


private fun initModelVariables(
    tf: Ops,
    weight: Variable<Float>,
    bias: Variable<Float>
): Pair<Assign<Float>, Assign<Float>> {
    val weightInit = tf.assign(weight, tf.constant(1f))
    val biasInit = tf.assign(bias, tf.constant(1f))
    return Pair(weightInit, biasInit)
}

private fun defineModelVariables(tf: Ops): Pair<Variable<Float>, Variable<Float>> {
    val weights: Variable<Float> =
        tf.variable(Shape.scalar(), Float::class.javaObjectType)
    val bias: Variable<Float> =
        tf.variable(Shape.scalar(), Float::class.javaObjectType)
    return Pair(weights, bias)
}

private fun definePlaceholders(tf: Ops): Pair<Placeholder<Float>, Placeholder<Float>> {
    val X = tf.placeholder(Float::class.javaObjectType, Placeholder.shape(Shape.scalar()))
    val Y = tf.placeholder(Float::class.javaObjectType, Placeholder.shape(Shape.scalar()))
    return Pair(X, Y)
}

