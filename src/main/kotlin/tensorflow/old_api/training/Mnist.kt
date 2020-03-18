package tensorflow.old_api.training

import org.tensorflow.*
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Assign
import org.tensorflow.op.core.Placeholder
import org.tensorflow.op.core.Variable
import org.tensorflow.op.math.Mean
import org.tensorflow.op.nn.Softmax
import org.tensorflow.op.train.ApplyGradientDescent
import tensorflow.old_api.training.util.ImageBatch
import tensorflow.old_api.training.util.ImageDataset

private const val VALIDATION_SIZE = 0
private const val TRAINING_BATCH_SIZE = 100

fun main() {
    val dataset =
        ImageDataset.create(VALIDATION_SIZE)

    Graph().use { graph ->
        val tf = Ops.create(graph)

        val (images, labels) = definePlaceholders(tf)

        val (weights: Variable<Float>, biases: Variable<Float>) = defineModelVariables(tf)

        val (weightsInit, biasesInit) = initModelVariables(tf, weights, biases)
        
        val (softmax, weightGradientDescent, biasGradientDescent) = buildTheGraph(tf, images, weights, biases, labels)

        Session(graph).use { session ->
            // Initialize graph variables
            session.runner()
                .addTarget(weightsInit)
                .addTarget(biasesInit)
                .run()

            train(dataset, session, weightGradientDescent, biasGradientDescent, images, labels)

            val accuracy = metricGraph(tf, softmax, labels)

            evaluateTheTestDataset(dataset, session, accuracy, images, labels)
        }
    }
}

private fun buildTheGraph(
    tf: Ops,
    images: Placeholder<Float>,
    weights: Variable<Float>,
    biases: Variable<Float>,
    labels: Placeholder<Float>
): Triple<Softmax<Float>, ApplyGradientDescent<Float>, ApplyGradientDescent<Float>> {
    val softmax = tf.nn.softmax(
        tf.math.add(
            tf.linalg.matMul(images, weights),
            biases
        )
    )
    val crossEntropy = tf.math.mean(
        tf.math.neg(
            tf.reduceSum(
                tf.math.mul(labels, tf.math.log(softmax)),
                constArray(tf, 1)
            )
        ), constArray(tf, 0)
    )

    val gradients = tf.gradients(crossEntropy, listOf(weights, biases))
    val alpha = tf.constant(0.2f)
    val weightGradientDescent =
        tf.train.applyGradientDescent(weights, alpha, gradients.dy<Float>(0))
    val biasGradientDescent =
        tf.train.applyGradientDescent(biases, alpha, gradients.dy<Float>(1))
    return Triple(softmax, weightGradientDescent, biasGradientDescent)
}

private fun metricGraph(
    tf: Ops,
    softmax: Softmax<Float>?,
    labels: Placeholder<Float>
): Mean<Float>? {
    val predicted: Operand<Long> = tf.math.argMax(softmax, tf.constant(1))
    val expected: Operand<Long> = tf.math.argMax(labels, tf.constant(1))

    val accuracy = tf.math.mean(
        tf.dtypes.cast(
            tf.math.equal<Long>(predicted, expected),
            Float::class.javaObjectType
        ), constArray(tf, 0)
    )
    return accuracy
}

private fun train(
    dataset: ImageDataset,
    session: Session,
    weightGradientDescent: ApplyGradientDescent<Float>?,
    biasGradientDescent: ApplyGradientDescent<Float>?,
    images: Placeholder<Float>,
    labels: Placeholder<Float>
) {
    // Train the graph
    val batchIter: ImageDataset.ImageBatchIterator = dataset.trainingBatchIterator(
        TRAINING_BATCH_SIZE
    )
    while (batchIter.hasNext()) {
        val batch: ImageBatch = batchIter.next()
        Tensor.create(batch.shape(784), batch.images()).use { batchImages ->
            Tensor.create(batch.shape(10), batch.labels()).use { batchLabels ->
                session.runner()
                    .addTarget(weightGradientDescent)
                    .addTarget(biasGradientDescent)
                    .feed(images.asOutput(), batchImages)
                    .feed(labels.asOutput(), batchLabels)
                    .run()
            }
        }
    }
}

private fun evaluateTheTestDataset(
    dataset: ImageDataset,
    session: Session,
    accuracy: Mean<Float>?,
    images: Placeholder<Float>,
    labels: Placeholder<Float>
) {
    val testBatch: ImageBatch = dataset.testBatch()
    Tensor.create(testBatch.shape(784), testBatch.images()).use { testImages ->
        Tensor.create(testBatch.shape(10), testBatch.labels()).use { testLabels ->
            session.runner()
                .fetch(accuracy)
                .feed(images.asOutput(), testImages)
                .feed(labels.asOutput(), testLabels)
                .run()[0].use { value -> println("Accuracy: " + value.floatValue()) }
        }
    }
}

private fun initModelVariables(
    tf: Ops,
    weights: Variable<Float>,
    biases: Variable<Float>
): Pair<Assign<Float>, Assign<Float>> {
    val weightsInit =
        tf.assign(weights, tf.zeros(constArray(tf, 784, 10), Float::class.javaObjectType))
    val biasesInit = tf.assign(biases, tf.zeros(constArray(tf, 10), Float::class.javaObjectType))
    return Pair(weightsInit, biasesInit)
}

private fun defineModelVariables(tf: Ops): Pair<Variable<Float>, Variable<Float>> {
    val weights: Variable<Float> =
        tf.variable(Shape.make(784, 10), Float::class.javaObjectType)
    val biases: Variable<Float> =
        tf.variable(Shape.make(10), Float::class.javaObjectType)
    return Pair(weights, biases)
}

private fun definePlaceholders(tf: Ops): Pair<Placeholder<Float>, Placeholder<Float>> {
    val images = tf.placeholder(
        Float::class.javaObjectType,
        Placeholder.shape(Shape.make(-1, 784))
    )

    val labels =
        tf.placeholder(Float::class.javaObjectType)
    return Pair(images, labels)
}

fun constArray(tf: Ops, vararg i: Int): Operand<Int> {
    return tf.constant(i)
}
