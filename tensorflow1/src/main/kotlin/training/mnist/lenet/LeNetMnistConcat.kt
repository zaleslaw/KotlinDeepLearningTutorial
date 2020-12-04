package training.mnist.lenet

import inference.printTFGraph
import org.tensorflow.*
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.*
import org.tensorflow.op.math.Add
import org.tensorflow.op.math.Mean
import org.tensorflow.op.nn.Conv2d
import org.tensorflow.op.nn.MaxPool
import org.tensorflow.op.nn.Relu
import org.tensorflow.op.nn.Softmax
import org.tensorflow.op.random.TruncatedNormal
import org.tensorflow.op.train.ApplyGradientDescent
import training.mnist.constArray
import training.util.ImageBatch
import training.util.ImageDataset

// Hyper-parameters
private const val LEARNING_RATE = 0.2f
private const val EPOCHS = 10
private const val TRAINING_BATCH_SIZE = 500

// Image pre-processing constants
private const val NUM_LABELS = 10L
private const val NUM_CHANNELS = 1L
private const val IMAGE_SIZE = 28L

private const val VALIDATION_SIZE = 0
private const val SEED = 12L
private const val PADDING_TYPE = "SAME"

// Tensor names
private const val INPUT_NAME = "input"
private const val OUTPUT_NAME = "output"
private const val TRAINING_LOSS = "training_loss"

fun main() {
    val dataset = ImageDataset.create(VALIDATION_SIZE)

    Graph().use { graph ->
        val tf = Ops.create(graph)

        // Define placeholders
        val (images, labels) = placeholders(tf)

        val (logits, variables, initVariables) = LeNet(tf, images)

        // Predicted outputs
        val prediction = predictionFunction(tf, logits)

        val loss = lossFunction(tf, logits, labels)

        // Define gradients
        val learningRate = tf.constant(LEARNING_RATE)

        val gradients = gradients(tf, loss, variables, learningRate)

        printTFGraph(graph)

        Session(graph).use { session ->

            // Initialize graph variables
            initializeGraphVariables(
                session,
                initVariables
            )

            // Train the graph
            train(
                dataset,
                session,
                gradients,
                images,
                labels
            )

            val accuracy = metric(tf, prediction, labels)

            evaluateTheTestDataset(dataset, session, accuracy, images, labels)
        }
    }
}

private fun predictionFunction(
    tf: Ops,
    logits: Add<Float>
) = tf.withName(OUTPUT_NAME).nn.softmax(logits)

private fun LeNet(
    tf: Ops,
    images: Placeholder<Float>
): Triple<Add<Float>, List<Variable<Float>>, List<Assign<Float>>> {
    // First conv layer
    val (conv1Weights: Variable<Float>, conv1WeightsInit, conv1) = conv1Weights(tf, images)

    val (conv1Biases: Variable<Float>, conv1BiasesInit) = conv1Biases(tf)

    val relu1 = relu(tf, conv1, conv1Biases)

    // First pooling layer
    val pool1 = maxPool(tf, relu1)

    // Second conv layer
    val (conv2Weights: Variable<Float>, conv2WeightsInit, conv2) = conv2Weights(tf, pool1)

    val (conv2Biases: Variable<Float>, conv2BiasesInit) = conv2Biases(tf)

    val relu2 = relu(tf, conv2, conv2Biases)

    // Second pooling layer
    val pool2 = maxPool(tf, relu2)

    // Flatten inputs
    val flatten = flatten(tf, pool2)


    // First conv layer
    val (conv11Weights: Variable<Float>, conv11WeightsInit, conv11) = conv1Weights(tf, images)

    val (conv11Biases: Variable<Float>, conv11BiasesInit) = conv1Biases(tf)

    val relu11 = relu(tf, conv11, conv11Biases)

    // First pooling layer
    val pool11 = maxPool(tf, relu11)

    // Second conv layer
    val (conv21Weights: Variable<Float>, conv21WeightsInit, conv21) = conv2Weights(tf, pool11)

    val (conv21Biases: Variable<Float>, conv21BiasesInit) = conv2Biases(tf)

    val relu21 = relu(tf, conv21, conv21Biases)

    // Second pooling layer
    val pool21 = maxPool(tf, relu21)

    // Flatten inputs
    val flatten1 = flatten(tf, pool21)

    val commonFlatten = tf.concat(mutableListOf(flatten, flatten1), tf.constant(-1))

    // Fully connected layer
    val (fc1Weights: Variable<Float>, fc1WeightsInit) = fc1Weights(tf)

    val (fc1Biases: Variable<Float>, fc1BiasesInit) = fc1Biases(tf)

    val relu3 = tf.nn.relu(tf.math.add(tf.linalg.matMul(commonFlatten, fc1Weights), fc1Biases))

    // Softmax layer
    val (fc2Weights: Variable<Float>, fc2WeightsInit) = fc2Weights(tf)

    val (fc2Biases: Variable<Float>, fc2BiasesInit) = fc2Biases(tf)

    val logits = tf.math.add(tf.linalg.matMul(relu3, fc2Weights), fc2Biases)

    val variables =
        listOf(conv1Weights, conv1Biases, conv2Weights, conv2Biases, fc1Weights, fc1Biases, fc2Weights, fc2Biases)

    val initVariables = listOf(
        conv1WeightsInit,
        conv1BiasesInit,
        conv2WeightsInit,
        conv2BiasesInit,
        fc1WeightsInit,
        fc1BiasesInit,
        fc2WeightsInit,
        fc2BiasesInit
    )
    return Triple(logits, variables, initVariables)
}

private fun conv1Weights(
    tf: Ops,
    images: Placeholder<Float>
): Triple<Variable<Float>, Assign<Float>, Conv2d<Float>> {
    // Generate random data to fill the weight matrix
    val conv1Shape = longArrayOf(5, 5, NUM_CHANNELS, 32)
    val truncatedNormal = truncatedNormal(tf, conv1Shape)

    val conv1Weights: Variable<Float> =
        tf.variable(Shape.make(5L, 5L, NUM_CHANNELS, 32), Float::class.javaObjectType)

    val conv1WeightsInit = tf.assign(conv1Weights, tf.math.mul(truncatedNormal, tf.constant(0.1f)))

    val conv1 = tf.nn.conv2d(
        images, conv1Weights, mutableListOf(1L, 1L, 1L, 1L),
        PADDING_TYPE
    );
    return Triple(conv1Weights, conv1WeightsInit, conv1)
}

private fun conv1Biases(tf: Ops): Pair<Variable<Float>, Assign<Float>> {
    val conv1Biases: Variable<Float> = tf.variable(Shape.make(32), Float::class.javaObjectType)

    val conv1BiasesInit = tf.assign(
        conv1Biases, tf.zeros(
            constArray(
                tf,
                32
            ), Float::class.javaObjectType
        )
    )
    return Pair(conv1Biases, conv1BiasesInit)
}

private fun conv2Weights(
    tf: Ops,
    pool1: MaxPool<Float>?
): Triple<Variable<Float>, Assign<Float>, Conv2d<Float>> {
    val conv2Shape = longArrayOf(5, 5, 32, 64)
    val truncatedNormal2 = truncatedNormal(tf, conv2Shape)

    val conv2Weights: Variable<Float> =
        tf.variable(Shape.make(5, 5, 32, 64), Float::class.javaObjectType)

    val conv2WeightsInit = tf.assign(conv2Weights, tf.math.mul(truncatedNormal2, tf.constant(0.1f)))

    val conv2 = tf.nn.conv2d(
        pool1, conv2Weights, mutableListOf(1L, 1L, 1L, 1L),
        PADDING_TYPE
    );
    return Triple(conv2Weights, conv2WeightsInit, conv2)
}

private fun conv2Biases(tf: Ops): Pair<Variable<Float>, Assign<Float>> {
    val conv2Biases: Variable<Float> = tf.variable(Shape.make(64), Float::class.javaObjectType)

    val conv2BiasesInit = tf.assign(
        conv2Biases, tf.zeros(
            constArray(
                tf,
                64
            ), Float::class.javaObjectType
        )
    )
    return Pair(conv2Biases, conv2BiasesInit)
}

private fun fc1Weights(tf: Ops): Pair<Variable<Float>, Assign<Float>> {
    val fc1Shape = longArrayOf(IMAGE_SIZE * IMAGE_SIZE * 4 * 2, 512)

    val truncatedNormal3 = truncatedNormal(tf, fc1Shape)

    val fc1Weights: Variable<Float> =
        tf.variable(Shape.make(IMAGE_SIZE * IMAGE_SIZE * 4 * 2, 512), Float::class.javaObjectType)

    val fc1WeightsInit = tf.assign(fc1Weights, tf.math.mul(truncatedNormal3, tf.constant(0.1f)))
    return Pair(fc1Weights, fc1WeightsInit)
}

private fun fc1Biases(tf: Ops): Pair<Variable<Float>, Assign<Float>> {
    val fc1Biases: Variable<Float> = tf.variable(Shape.make(512), Float::class.javaObjectType)

    val fc1BiasesInit = tf.assign(fc1Biases, tf.fill(tf.constant(intArrayOf(512)), tf.constant(0.1f)))
    return Pair(fc1Biases, fc1BiasesInit)
}

private fun fc2Biases(tf: Ops): Pair<Variable<Float>, Assign<Float>> {
    val fc2Biases: Variable<Float> = tf.variable(Shape.make(NUM_LABELS), Float::class.javaObjectType)

    val fc2BiasesInit =
        tf.assign(fc2Biases, tf.fill(tf.constant(intArrayOf(NUM_LABELS.toInt())), tf.constant(0.1f)))
    return Pair(fc2Biases, fc2BiasesInit)
}

private fun fc2Weights(tf: Ops): Pair<Variable<Float>, Assign<Float>> {
    val fc2Shape = longArrayOf(512, NUM_LABELS)
    val truncatedNormal4 = truncatedNormal(tf, fc2Shape)

    val fc2Weights: Variable<Float> =
        tf.variable(Shape.make(512, NUM_LABELS), Float::class.javaObjectType)

    val fc2WeightsInit = tf.assign(fc2Weights, tf.math.mul(truncatedNormal4, tf.constant(0.1f)))
    return Pair(fc2Weights, fc2WeightsInit)
}

private fun flatten(tf: Ops, pool2: MaxPool<Float>?): Reshape<Float>? {
    val slice: Slice<Int> = tf.slice(
        tf.shape(pool2),
        tf.constant(intArrayOf(0)),
        tf.constant(intArrayOf(1))
    )

    val mutableListOf: MutableList<Operand<Int>> = mutableListOf(slice, tf.constant(intArrayOf(-1)))

    return tf.reshape(
        pool2,
        tf.concat(
            mutableListOf,
            tf.constant(0)
        )
    )
}

private fun relu(
    tf: Ops,
    conv2: Conv2d<Float>?,
    conv2Biases: Variable<Float>
) = tf.nn.relu(tf.nn.biasAdd(conv2, conv2Biases))

private fun maxPool(
    tf: Ops,
    relu1: Relu<Float>?
): MaxPool<Float>? {
    return tf.nn.maxPool(
        relu1,
        tf.constant(intArrayOf(1, 2, 2, 1)),
        tf.constant(intArrayOf(1, 2, 2, 1)),
        PADDING_TYPE
    )
}

private fun truncatedNormal(
    tf: Ops,
    fc2Shape: LongArray
): TruncatedNormal<Float>? {
    return tf.random.truncatedNormal(
        tf.constant(fc2Shape),
        Float::class.javaObjectType,
        TruncatedNormal.seed(SEED)
    )
}

private fun train(
    dataset: ImageDataset,
    session: Session,
    gradients: List<ApplyGradientDescent<Float>>,
    images: Placeholder<Float>,
    labels: Placeholder<Float>
) {
    for (i in 1..EPOCHS) {
        val batchIter: ImageDataset.ImageBatchIterator = dataset.trainingBatchIterator(
            TRAINING_BATCH_SIZE
        )

        while (batchIter.hasNext()) {
            val batch: ImageBatch = batchIter.next()
            Tensor.create(
                longArrayOf(
                    batch.size().toLong(),
                    IMAGE_SIZE,
                    IMAGE_SIZE,
                    NUM_CHANNELS
                ),
                batch.images()
            ).use { batchImages ->
                Tensor.create(longArrayOf(batch.size().toLong(), 10), batch.labels()).use { batchLabels ->
                    val lossValue = session.runner()
                        .addTarget(gradients[0])
                        .addTarget(gradients[1])
                        .addTarget(gradients[2])
                        .addTarget(gradients[3])
                        .addTarget(gradients[4])
                        .addTarget(gradients[5])
                        .addTarget(gradients[6])
                        .addTarget(gradients[7])
                        .feed(images.asOutput(), batchImages)
                        .feed(labels.asOutput(), batchLabels)
                        .fetch(TRAINING_LOSS)
                        .run()[0].floatValue()
                    println("epochs: $i lossValue: $lossValue")
                }
            }
        }
    }
}

private fun initializeGraphVariables(session: Session, initVariables: List<Assign<Float>>) {
    session.runner()
        .addTarget(initVariables[0])
        .addTarget(initVariables[1])
        .addTarget(initVariables[2])
        .addTarget(initVariables[3])
        .addTarget(initVariables[4])
        .addTarget(initVariables[5])
        .addTarget(initVariables[6])
        .addTarget(initVariables[7])
        .run()
}

private fun gradients(
    tf: Ops,
    loss: Mean<Float>?,
    variables: List<Variable<Float>>,
    learningRate: Constant<Float>?
): List<ApplyGradientDescent<Float>> {
    val gradients = tf.gradients(loss, variables)

    // Set up the SGD for all variables
    val conv1WeightsGD = tf.train.applyGradientDescent(variables[0], learningRate, gradients.dy(0))
    val conv1BiasesGD = tf.train.applyGradientDescent(variables[1], learningRate, gradients.dy(1))
    val conv2WeightsGD = tf.train.applyGradientDescent(variables[2], learningRate, gradients.dy(2))
    val conv2BiasesGD = tf.train.applyGradientDescent(variables[3], learningRate, gradients.dy(3))
    val fc1WeightsGD = tf.train.applyGradientDescent(variables[4], learningRate, gradients.dy(4))
    val fc1BiasesGD = tf.train.applyGradientDescent(variables[5], learningRate, gradients.dy(5))
    val fc2WeightsGD = tf.train.applyGradientDescent(variables[6], learningRate, gradients.dy(6))
    val fc2BiasesGD = tf.train.applyGradientDescent(variables[7], learningRate, gradients.dy(7))

    return listOf(
        conv1WeightsGD,
        conv1BiasesGD,
        conv2WeightsGD,
        conv2BiasesGD,
        fc1WeightsGD,
        fc1BiasesGD,
        fc2WeightsGD,
        fc2BiasesGD
    )
}

private fun lossFunction(
    tf: Ops,
    logits: Add<Float>?,
    labels: Placeholder<Float>
): Mean<Float>? {
    val batchLoss = tf.nn.softmaxCrossEntropyWithLogits(logits, labels)

    val loss = tf.withName(TRAINING_LOSS).math.mean(batchLoss.loss(), tf.constant(0))
    return loss
}

private fun metric(
    tf: Ops,
    prediction: Softmax<Float>?,
    labels: Placeholder<Float>
): Mean<Float>? {
    // Evaluate the model
    val predicted: Operand<Long> = tf.math.argMax(prediction, tf.constant(1))
    val expected: Operand<Long> = tf.math.argMax(labels, tf.constant(1))

    // Define multi-classification metric
    val accuracy = tf.math.mean(
        tf.dtypes.cast(
            tf.math.equal(predicted, expected),
            Float::class.javaObjectType
        ), constArray(tf, 0)
    )
    return accuracy
}

private fun evaluateTheTestDataset(
    dataset: ImageDataset,
    session: Session,
    accuracy: Mean<Float>?,
    images: Placeholder<Float>,
    labels: Placeholder<Float>
) {
    val testBatch: ImageBatch = dataset.testBatch()
    Tensor.create(
        longArrayOf(
            testBatch.size().toLong(),
            IMAGE_SIZE,
            IMAGE_SIZE,
            NUM_CHANNELS
        ),
        testBatch.images()
    ).use { testImages ->
        Tensor.create(testBatch.shape(10), testBatch.labels()).use { testLabels ->
            session.runner()
                .fetch(accuracy)
                .feed(images.asOutput(), testImages)
                .feed(labels.asOutput(), testLabels)
                .run()[0].use { value -> println("Accuracy: " + value.floatValue()) }
        }
    }
}


private fun placeholders(tf: Ops): Pair<Placeholder<Float>, Placeholder<Float>> {
    val images = tf.withName(INPUT_NAME).placeholder(
        Float::class.javaObjectType,
        Placeholder.shape(
            Shape.make(
                -1,
                IMAGE_SIZE,
                IMAGE_SIZE,
                NUM_CHANNELS
            )
        )
    )

    val labels = tf.placeholder(Float::class.javaObjectType)
    return Pair(images, labels)
}
