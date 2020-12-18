import datasets.mnist.MnistDataset
import org.tensorflow.Graph
import org.tensorflow.Operand
import org.tensorflow.Session
import org.tensorflow.framework.optimizers.Adam
import org.tensorflow.framework.optimizers.Optimizer
import org.tensorflow.ndarray.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Placeholder
import org.tensorflow.op.core.Reshape
import org.tensorflow.op.core.Slice
import org.tensorflow.op.core.Variable
import org.tensorflow.op.math.Add
import org.tensorflow.op.math.Mean
import org.tensorflow.op.nn.Conv2d
import org.tensorflow.op.nn.MaxPool
import org.tensorflow.op.nn.Relu
import org.tensorflow.op.nn.Softmax
import org.tensorflow.op.random.TruncatedNormal
import org.tensorflow.types.TFloat32
import org.tensorflow.types.TInt32
import org.tensorflow.types.TInt64
import org.tensorflow.types.TUint8


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
private const val INIT = "init"
private const val TRAIN = "train"
private const val PIXEL_DEPTH = 255


// Fashion MNIST dataset paths
const val TRAINING_IMAGES_ARCHIVE = "fashionmnist/train-images-idx3-ubyte.gz"
const val TRAINING_LABELS_ARCHIVE = "fashionmnist/train-labels-idx1-ubyte.gz"
const val TEST_IMAGES_ARCHIVE = "fashionmnist/t10k-images-idx3-ubyte.gz"
const val TEST_LABELS_ARCHIVE = "fashionmnist/t10k-labels-idx1-ubyte.gz"

fun main() {
    val dataset = MnistDataset.create(
        0,
        TRAINING_IMAGES_ARCHIVE,
        TRAINING_LABELS_ARCHIVE,
        TEST_IMAGES_ARCHIVE,
        TEST_LABELS_ARCHIVE
    )

    Graph().use { graph ->
        val tf = Ops.create(graph)

        // Define placeholders
        val (images, labels) = placeholders(tf)

        val logits = LeNet(tf, images, labels)

        // Predicted outputs
        val prediction = predictionFunction(tf, logits)

        val loss = lossFunction(tf, logits, labels)

        Session(graph).use { session ->
            tf.init()
            val optimizer: Optimizer = Adam(graph, 0.001f, 0.9f, 0.999f, 1e-8f)

            optimizer.minimize(loss, TRAIN)

            // Initialises the parameters.
            session.runner().addTarget(INIT).run()

            // Train the graph
            train(
                dataset,
                session,
                images,
                labels
            )


            //val accuracy = metric(tf, prediction, labels)

            //evaluateTheTestDataset(dataset, session, accuracy, images, labels)
        }
    }
}

private fun predictionFunction(
    tf: Ops,
    logits: Add<TFloat32>
) = tf.withName(OUTPUT_NAME).nn.softmax(logits)

private fun LeNet(
    tf: Ops,
    images: Placeholder<TUint8>,
    labels: Placeholder<TUint8>
): Add<TFloat32> {
    // Scaling the features

    // Scaling the features
    val centeringFactor = tf.constant(PIXEL_DEPTH / 2.0f)
    val scalingFactor = tf.constant(PIXEL_DEPTH.toFloat())
    val scaledInput: Operand<TFloat32> = tf.math
        .div(
            tf.math.sub(tf.dtypes.cast(images, TFloat32.DTYPE), centeringFactor),
            scalingFactor
        )


    // First conv layer
    val (conv1Weights: Variable<TFloat32>, conv1) = conv1Weights(tf, scaledInput)

    val conv1Biases: Variable<TFloat32> = conv1Biases(tf)

    val relu1 = relu(tf, conv1, conv1Biases)

    // First pooling layer
    val pool1 = maxPool(tf, relu1)

    // Second conv layer
    val (conv2Weights: Variable<TFloat32>, conv2) = conv2Weights(tf, pool1)

    val conv2Biases: Variable<TFloat32> = conv2Biases(tf)

    val relu2 = relu(tf, conv2, conv2Biases)

    // Second pooling layer
    val pool2 = maxPool(tf, relu2)

    // Flatten inputs
    val flatten = flatten(tf, pool2)


    // First conv layer
    val (conv11Weights: Variable<TFloat32>, conv11) = conv1Weights(tf, scaledInput)

    val conv11Biases: Variable<TFloat32> = conv1Biases(tf)

    val relu11 = relu(tf, conv11, conv11Biases)

    // First pooling layer
    val pool11 = maxPool(tf, relu11)

    // Second conv layer
    val (conv21Weights: Variable<TFloat32>, conv21) = conv2Weights(tf, pool11)

    val conv21Biases: Variable<TFloat32> = conv2Biases(tf)

    val relu21 = relu(tf, conv21, conv21Biases)

    // Second pooling layer
    val pool21 = maxPool(tf, relu21)

    // Flatten inputs
    val flatten1 = flatten(tf, pool21)

    val commonFlatten = tf.concat(mutableListOf(flatten, flatten1), tf.constant(-1))

    // Fully connected layer
    val fc1Weights: Variable<TFloat32> = fc1Weights(tf)

    val fc1Biases: Variable<TFloat32> = fc1Biases(tf)

    val relu3 = tf.nn.relu(tf.math.add(tf.linalg.matMul(commonFlatten, fc1Weights), fc1Biases))

    // Softmax layer
    val fc2Weights: Variable<TFloat32> = fc2Weights(tf)

    val fc2Biases: Variable<TFloat32> = fc2Biases(tf)

    val logits = tf.math.add(tf.linalg.matMul(relu3, fc2Weights), fc2Biases)

    return logits
}

private fun conv1Weights(
    tf: Ops,
    images: Operand<TFloat32>
): Pair<Variable<TFloat32>, Conv2d<TFloat32>> {
    // Generate random data to fill the weight matrix
    val conv1Shape = longArrayOf(5, 5, NUM_CHANNELS, 32)
    val truncatedNormal = truncatedNormal(tf, conv1Shape)

    val conv1Weights: Variable<TFloat32> =
        tf.variable(tf.math.mul(truncatedNormal, tf.constant(0.1f)))

    val conv1 = tf.nn.conv2d(
        images, conv1Weights, mutableListOf(1L, 1L, 1L, 1L),
        PADDING_TYPE
    );
    return Pair(conv1Weights, conv1)
}

private fun conv1Biases(tf: Ops): Variable<TFloat32> {
    val conv1Biases: Variable<TFloat32> = tf.variable(
        tf.zeros(
            constArray(
                tf,
                32
            ), TFloat32.DTYPE
        )
    )

    return conv1Biases
}

private fun conv2Weights(
    tf: Ops,
    pool1: MaxPool<TFloat32>?
): Pair<Variable<TFloat32>, Conv2d<TFloat32>> {
    val conv2Shape = longArrayOf(5, 5, 32, 64)
    val truncatedNormal2 = truncatedNormal(tf, conv2Shape)

    val conv2Weights: Variable<TFloat32> =
        tf.variable(tf.math.mul(truncatedNormal2, tf.constant(0.1f)))


    val conv2 = tf.nn.conv2d(
        pool1, conv2Weights, mutableListOf(1L, 1L, 1L, 1L),
        PADDING_TYPE
    );
    return Pair(conv2Weights, conv2)
}

private fun conv2Biases(tf: Ops): Variable<TFloat32> {
    val conv2Biases: Variable<TFloat32> = tf.variable(
        tf.zeros(
            constArray(
                tf,
                64
            ), TFloat32.DTYPE
        )
    )

    return conv2Biases
}

private fun fc1Weights(tf: Ops): Variable<TFloat32> {
    val fc1Shape = longArrayOf(IMAGE_SIZE * IMAGE_SIZE * 4 * 2, 512)

    val truncatedNormal3 = truncatedNormal(tf, fc1Shape)

    val fc1Weights: Variable<TFloat32> =
        tf.variable(tf.math.mul(truncatedNormal3, tf.constant(0.1f)))

    return fc1Weights
}

private fun fc1Biases(tf: Ops): Variable<TFloat32> {
    val fc1Biases: Variable<TFloat32> = tf.variable(tf.fill(tf.constant(intArrayOf(512)), tf.constant(0.1f)))
    return fc1Biases
}

private fun fc2Biases(tf: Ops): Variable<TFloat32> {
    val fc2Biases: Variable<TFloat32> = tf.variable(
        tf.fill(
            tf.constant(intArrayOf(NUM_LABELS.toInt())), tf.constant(
                0.1f
            )
        )
    )
    return fc2Biases
}

private fun fc2Weights(tf: Ops): Variable<TFloat32> {
    val fc2Shape = longArrayOf(512, NUM_LABELS)
    val truncatedNormal4 = truncatedNormal(tf, fc2Shape)

    val fc2Weights: Variable<TFloat32> =
        tf.variable(tf.math.mul(truncatedNormal4, tf.constant(0.1f)))
    return fc2Weights
}

private fun flatten(tf: Ops, pool2: MaxPool<TFloat32>?): Reshape<TFloat32>? {
    val slice: Slice<TInt32> = tf.slice(
        tf.shape(pool2),
        tf.constant(intArrayOf(0)),
        tf.constant(intArrayOf(1))
    )

    val mutableListOf: MutableList<Operand<TInt32>> = mutableListOf(slice, tf.constant(intArrayOf(-1)))

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
    conv2: Conv2d<TFloat32>?,
    conv2Biases: Variable<TFloat32>
) = tf.nn.relu(tf.nn.biasAdd(conv2, conv2Biases))

private fun maxPool(
    tf: Ops,
    relu1: Relu<TFloat32>?
): MaxPool<TFloat32>? {
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
): TruncatedNormal<TFloat32>? {
    return tf.random.truncatedNormal(
        tf.constant(fc2Shape),
        TFloat32.DTYPE,
        TruncatedNormal.seed(SEED)
    )
}

private fun train(
    dataset: MnistDataset,
    session: Session,
    images: Placeholder<TUint8>,
    labels: Placeholder<TUint8>
) {
    for (i in 1..EPOCHS) {
        for (trainingBatch in dataset.trainingBatches(TRAINING_BATCH_SIZE)) {
            TUint8.tensorOf(trainingBatch.images()).use { batchImages ->
                TUint8.tensorOf(trainingBatch.labels()).use { batchLabels ->
                    val lossValue = session.runner()
                        .feed(images.asOutput(), batchImages)
                        .feed(labels.asOutput(), batchLabels)
                        .addTarget(TRAIN)
                        .fetch(TRAINING_LOSS)
                        .run()[0].expect(TFloat32.DTYPE)

                    println("epochs: $i lossValue: $lossValue")

                }
            }
        }
    }
}


private fun lossFunction(
    tf: Ops,
    logits: Add<TFloat32>?,
    labels: Placeholder<TUint8>
): Mean<TFloat32>? {

    // Loss function & regularization
    val oneHot = tf
        .oneHot(labels, tf.constant(10), tf.constant(1.0f), tf.constant(0.0f))
    val batchLoss = tf.nn.raw
        .softmaxCrossEntropyWithLogits(logits, oneHot)

    return tf.withName(TRAINING_LOSS).math.mean(batchLoss.loss(), tf.constant(0))
}

private fun metric(
    tf: Ops,
    prediction: Softmax<TFloat32>?,
    labels: Placeholder<TFloat32>
): Mean<TFloat32>? {
    // Evaluate the model
    val predicted: Operand<TInt64> = tf.math.argMax(prediction, tf.constant(1))
    val expected: Operand<TInt64> = tf.math.argMax(labels, tf.constant(1))

    // Define multi-classification metric
    val accuracy = tf.math.mean(
        tf.dtypes.cast(
            tf.math.equal(predicted, expected),
            TFloat32.DTYPE
        ), constArray(tf, 0)
    )
    return accuracy
}

/*private fun evaluateTheTestDataset(
    dataset: MnistDataset,
    session: Session,
    accuracy: Mean<TFloat32>?,
    images: Placeholder<TFloat32>,
    labels: Placeholder<TFloat32>
) {
    val testBatch: ImageBatch = dataset.testBatch()

    for (trainingBatch in dataset.testBatches(TRAINING_BATCH_SIZE)) {
        TUint8.tensorOf(
            trainingBatch.images()
        ).use { batchImages ->
            TUint8.tensorOf(trainingBatch.labels()).use { batchLabels ->
                val lossValue = session.runner()
                    .feed(images.asOutput(), batchImages)
                    .feed(labels.asOutput(), batchLabels)
                    .addTarget(TRAIN)
                    .fetch(TRAINING_LOSS)
                    .run()[0].expect(TFloat32.DTYPE)) {
                println("epochs: $i lossValue: $lossValue")
            }

            }
        }
    }



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
                .run()[0].use { value -> println("Accuracy: " + value.TFloat32Value()) }
        }
    }
}*/


private fun placeholders(tf: Ops): Pair<Placeholder<TUint8>, Placeholder<TUint8>> {
    val images = tf.withName(INPUT_NAME).placeholder(
        TUint8.DTYPE,
        Placeholder.shape(
            Shape.of(
                -1,
                IMAGE_SIZE,
                IMAGE_SIZE,
                NUM_CHANNELS
            )
        )
    )

    val labels = tf.placeholder(TUint8.DTYPE)
    return Pair(images, labels)
}

fun constArray(tf: Ops, vararg i: Int): Operand<TInt32> {
    return tf.constant(i)
}
