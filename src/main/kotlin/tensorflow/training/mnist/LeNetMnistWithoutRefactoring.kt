package tensorflow.training.mnist

import org.tensorflow.*
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Placeholder
import org.tensorflow.op.core.Variable
import org.tensorflow.op.random.TruncatedNormal
import tensorflow.inference.printTFGraph
import tensorflow.training.util.ImageBatch
import tensorflow.training.util.ImageDataset

private const val PIXEL_DEPTH = 255f
private const val NUM_CHANNELS = 1L
private const val IMAGE_SIZE = 28L
private const val VALIDATION_SIZE = 0
private const val TRAINING_BATCH_SIZE = 100
private const val SEED = 123456789L


// The reference implementation from the new TF Java API
// https://github.com/tensorflow/java-models/blob/mnist/tensorflow-examples/src/main/java/org/tensorflow/model/examples/mnist/CnnMnist.java
fun main() {
    val dataset =
        ImageDataset.create(VALIDATION_SIZE)

    Graph().use { graph ->
        val tf = Ops.create(graph)

        // Define placeholders
        val images = tf.placeholder(
            Float::class.javaObjectType,
            Placeholder.shape(Shape.make(-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
        )

        val labels = tf.placeholder(Float::class.javaObjectType)


        val centeringFactor = tf.constant(PIXEL_DEPTH / 2.0f)
        val scalingFactor = tf.constant(PIXEL_DEPTH)

        val scaledInput = tf.math
            .div(
                tf.math
                    .sub(
                        tf.dtypes
                            .cast(images, Float::class.javaObjectType), centeringFactor
                    ),
                scalingFactor
            )


        // First conv layer

        val truncatedNormal = tf.random
            .truncatedNormal(
                tf.shape(v), Float::class.javaObjectType,
                TruncatedNormal.seed(SEED)
            )


        val conv1Weights: Variable<Float> =
            tf.variable(Shape.make(5L, 5L, NUM_CHANNELS, 32), Float::class.javaObjectType)

        val conv1WeightsInit = tf.assign(conv1Weights, truncatedNormal)


        // Define variables
        /*val weights: Variable<Float> =
            tf.variable(Shape.make(784, 10), Float::class.javaObjectType)
        val biases: Variable<Float> =
            tf.variable(Shape.make(10), Float::class.javaObjectType)*/

        // Init variables
        val weightsInit =
            tf.assign(weights, tf.zeros(constArray(tf, 784, 10), Float::class.javaObjectType))
        val biasesInit = tf.assign(biases, tf.zeros(constArray(tf, 10), Float::class.javaObjectType))

        val softmax = tf.nn.softmax(
            tf.math.add(
                tf.linalg.matMul(images, weights),
                biases
            )
        )

        // Define loss function
        val crossEntropy = tf.math.mean(
            tf.math.neg(
                tf.reduceSum(
                    tf.math.mul(labels, tf.math.log(softmax)),
                    constArray(tf, 1)
                )
            ), constArray(tf, 0)
        )

        // Define gradients
        val gradients = tf.gradients(crossEntropy, listOf(weights, biases))
        val alpha = tf.constant(0.2f)
        val weightGradientDescent =
            tf.train.applyGradientDescent(weights, alpha, gradients.dy<Float>(0))
        val biasGradientDescent =
            tf.train.applyGradientDescent(biases, alpha, gradients.dy<Float>(1))

        printTFGraph(graph)

        Session(graph).use { session ->
            // Initialize graph variables
            session.runner()
                .addTarget(weightsInit)
                .addTarget(biasesInit)
                .run()

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

            val predicted: Operand<Long> = tf.math.argMax(softmax, tf.constant(1))
            val expected: Operand<Long> = tf.math.argMax(labels, tf.constant(1))

            // Define multi-classification metric
            val accuracy = tf.math.mean(
                tf.dtypes.cast(
                    tf.math.equal(predicted, expected),
                    Float::class.javaObjectType
                ), constArray(tf, 0)
            )

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
    }
}