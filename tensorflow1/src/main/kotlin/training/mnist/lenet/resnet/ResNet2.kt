package training.mnist.lenet.resnet

import inference.printTFGraph
import org.tensorflow.*
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Placeholder
import org.tensorflow.op.core.Slice
import org.tensorflow.op.core.Variable
import org.tensorflow.op.random.TruncatedNormal
import training.mnist.constArray
import training.util.ImageBatch
import training.util.ImageDataset

// Hyper-parameters
private const val LEARNING_RATE = 0.2f
private const val EPOCHS = 3
private const val TRAINING_BATCH_SIZE = 1000

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


/** Accuracy: 0.9019 */

/**
 * Here we see
 *
 *     | ---- identity -- |                       | ---- identity -- |
 *  -- +- conv2d/conv2d --+ -> Relu -> MaxPool-> +- conv2d/conv2d --+ -> Relu -> MaxPool -> Flatten -> Dense Top
 */
fun main() {
    val dataset = ImageDataset.create(VALIDATION_SIZE)

    Graph().use { graph ->
        val tf = Ops.create(graph)

        // Define placeholders
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

        // First conv block

        // Generate random data to fill the weight matrix
        val truncatedNormal = tf.random.truncatedNormal(
            tf.constant(longArrayOf(5, 5, NUM_CHANNELS, 32)),
            Float::class.javaObjectType,
            TruncatedNormal.seed(SEED)
        )


        val conv11Weights: Variable<Float> =
            tf.variable(Shape.make(5L, 5L, NUM_CHANNELS, 32), Float::class.javaObjectType)

        val conv11WeightsInit = tf.assign(conv11Weights, tf.math.mul(truncatedNormal, tf.constant(0.1f)))

        val conv11 = tf.nn.conv2d(
            images, conv11Weights, mutableListOf(1L, 1L, 1L, 1L),
            PADDING_TYPE
        )

        val conv11Biases: Variable<Float> = tf.variable(Shape.make(32), Float::class.javaObjectType)

        val conv11BiasesInit = tf.assign(
            conv11Biases, tf.zeros(
                constArray(
                    tf,
                    32
                ), Float::class.javaObjectType
            )
        )

        val relu11 = tf.nn.relu(tf.nn.biasAdd(conv11, conv11Biases))

        val conv12Weights: Variable<Float> =
            tf.variable(Shape.make(5L, 5L, 32, 32), Float::class.javaObjectType)

        // Generate random data to fill the weight matrix
        val truncatedNormal2 = tf.random.truncatedNormal(
            tf.constant(longArrayOf(5, 5, 32, 32)),
            Float::class.javaObjectType,
            TruncatedNormal.seed(SEED)
        )

        val conv12WeightsInit = tf.assign(conv12Weights, tf.math.mul(truncatedNormal2, tf.constant(0.01f)))

        val conv12 = tf.nn.conv2d(
            relu11, conv12Weights, mutableListOf(1L, 1L, 1L, 1L),
            PADDING_TYPE
        )

        val conv12Biases: Variable<Float> = tf.variable(Shape.make(32), Float::class.javaObjectType)

        val conv12BiasesInit = tf.assign(
            conv12Biases, tf.zeros(
                constArray(
                    tf,
                    32
                ), Float::class.javaObjectType
            )
        )

        val relu12 = tf.nn.relu(tf.math.add(images, tf.nn.biasAdd(conv12, conv12Biases)))

        // First pooling layer
        val pool1 = tf.nn.maxPool(
            relu12,
            tf.constant(intArrayOf(1, 2, 2, 1)),
            tf.constant(intArrayOf(1, 2, 2, 1)),
            PADDING_TYPE
        )

        // Second conv block

        val conv21Weights: Variable<Float> =
            tf.variable(Shape.make(5, 5, 32, 32), Float::class.javaObjectType)

        // Generate random data to fill the weight matrix
        val truncatedNormal3 = tf.random.truncatedNormal(
            tf.constant(longArrayOf(5, 5, 32, 32)),
            Float::class.javaObjectType,
            TruncatedNormal.seed(SEED)
        )

        val conv21WeightsInit = tf.assign(conv21Weights, tf.math.mul(truncatedNormal3, tf.constant(0.1f)))

        val conv21 = tf.nn.conv2d(
            pool1, conv21Weights, mutableListOf(1L, 1L, 1L, 1L),
            PADDING_TYPE
        )

        val conv21Biases: Variable<Float> = tf.variable(Shape.make(32), Float::class.javaObjectType)

        val conv21BiasesInit = tf.assign(
            conv21Biases, tf.zeros(
                constArray(
                    tf,
                    32
                ), Float::class.javaObjectType
            )
        )

        val relu21 = tf.nn.relu(tf.nn.biasAdd(conv21, conv21Biases))


        val conv22Weights: Variable<Float> =
            tf.variable(Shape.make(5, 5, 32, 32), Float::class.javaObjectType)


        // Generate random data to fill the weight matrix
        val truncatedNormal4 = tf.random.truncatedNormal(
            tf.constant(longArrayOf(5, 5, 32, 32)),
            Float::class.javaObjectType,
            TruncatedNormal.seed(SEED)
        )
        val conv22WeightsInit = tf.assign(conv22Weights, tf.math.mul(truncatedNormal4, tf.constant(0.01f)))

        val conv22 = tf.nn.conv2d(
            relu21, conv22Weights, mutableListOf(1L, 1L, 1L, 1L),
            PADDING_TYPE
        )

        val conv22Biases: Variable<Float> = tf.variable(Shape.make(32), Float::class.javaObjectType)

        val conv22BiasesInit = tf.assign(
            conv22Biases, tf.zeros(
                constArray(
                    tf,
                    32
                ), Float::class.javaObjectType
            )
        )

        val relu22 = tf.nn.relu(tf.math.add(pool1, tf.nn.biasAdd(conv22, conv22Biases)))

        // Second pooling layer
        val pool2 = tf.nn.maxPool(
            relu22,
            tf.constant(intArrayOf(1, 2, 2, 1)),
            tf.constant(intArrayOf(1, 2, 2, 1)),
            PADDING_TYPE
        )

        // Flatten inputs
        val slice: Slice<Int> = tf.slice(
            tf.shape(pool2),
            tf.constant(intArrayOf(0)),
            tf.constant(intArrayOf(1))
        )

        val mutableListOf: MutableList<Operand<Int>> = mutableListOf(slice, tf.constant(intArrayOf(-1)))

        val flatten = tf.reshape(
            pool2,
            tf.concat(
                mutableListOf,
                tf.constant(0)
            )
        )

        // Fully connected layer
        val truncatedNormal5 = tf.random.truncatedNormal(
            tf.constant(longArrayOf(IMAGE_SIZE * IMAGE_SIZE * 2, 512)),
            Float::class.javaObjectType,
            TruncatedNormal.seed(SEED)
        )

        val fc1Weights: Variable<Float> =
            tf.variable(Shape.make(IMAGE_SIZE * IMAGE_SIZE * 2, 512), Float::class.javaObjectType)

        val fc1WeightsInit = tf.assign(fc1Weights, tf.math.mul(truncatedNormal5, tf.constant(0.1f)))

        val fc1Biases: Variable<Float> = tf.variable(Shape.make(512), Float::class.javaObjectType)

        val fc1BiasesInit = tf.assign(fc1Biases, tf.fill(tf.constant(intArrayOf(512)), tf.constant(0.1f)))

        val relu3 = tf.nn.relu(tf.math.add(tf.linalg.matMul(flatten, fc1Weights), fc1Biases))

        // Softmax layer
        val truncatedNormal6 = tf.random.truncatedNormal(
            tf.constant(longArrayOf(512, NUM_LABELS)),
            Float::class.javaObjectType,
            TruncatedNormal.seed(SEED)
        )

        val fc2Weights: Variable<Float> =
            tf.variable(Shape.make(512, NUM_LABELS), Float::class.javaObjectType)

        val fc2WeightsInit = tf.assign(fc2Weights, tf.math.mul(truncatedNormal6, tf.constant(0.1f)))

        val fc2Biases: Variable<Float> = tf.variable(Shape.make(NUM_LABELS), Float::class.javaObjectType)

        val fc2BiasesInit =
            tf.assign(fc2Biases, tf.fill(tf.constant(intArrayOf(NUM_LABELS.toInt())), tf.constant(0.1f)))

        val logits = tf.math.add(tf.linalg.matMul(relu3, fc2Weights), fc2Biases)

        // Predicted outputs
        val prediction = tf.withName(OUTPUT_NAME).nn.softmax(logits)

        val batchLoss = tf.nn.softmaxCrossEntropyWithLogits(logits, labels)

        val loss = tf.withName(TRAINING_LOSS).math.mean(batchLoss.loss(), tf.constant(0))

        // Define gradients
        val learningRate = tf.constant(LEARNING_RATE)

        val variables =
            listOf(
                conv11Weights,
                conv11Biases,
                conv21Weights,
                conv21Biases,
                conv12Weights,
                conv12Biases,
                conv22Weights,
                conv22Biases,
                fc1Weights,
                fc1Biases,
                fc2Weights,
                fc2Biases
            )

        val gradients = tf.gradients(loss, variables)

        // Set up the SGD for all variables
        val variablesGD = variables.mapIndexed { index, variable ->
            tf.train.applyGradientDescent(variable, learningRate, gradients.dy(index))
        }

        val variablesInit = listOf(
            conv11WeightsInit, conv11BiasesInit, conv21WeightsInit, conv21BiasesInit,
            conv12WeightsInit, conv12BiasesInit, conv22WeightsInit, conv22BiasesInit,
            fc1WeightsInit, fc1BiasesInit, fc2WeightsInit, fc2BiasesInit
        )

        printTFGraph(graph)

        Session(graph).use { session ->

            fun <T, E> T.applyF(f: T.(E) -> T, ls: Iterable<E>) = ls.fold(this, f)

            // Initialize graph variables
            session.runner()
                .applyF(Session.Runner::addTarget, variablesInit)
                .run()

            // Train the graph
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
                                .applyF(Session.Runner::addTarget, variablesGD)
                                .feed(images.asOutput(), batchImages)
                                .feed(labels.asOutput(), batchLabels)
                                .fetch(TRAINING_LOSS)
                                .run()[0].floatValue()
                            println("epochs: $i lossValue: $lossValue")
                        }
                    }
                }
            }

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
    }
}
