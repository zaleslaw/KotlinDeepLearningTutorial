import org.tensorflow.*
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Placeholder
import org.tensorflow.op.core.Variable
import java.util.*

private const val VALIDATION_SIZE = 0
private const val TRAINING_BATCH_SIZE = 100

fun main() {
    val dataset = ImageDataset.create(VALIDATION_SIZE)

    println(TensorFlow.version())
    Graph().use { graph ->

        val tf = Ops.create(graph)

        val images = tf.placeholder(
            Float::class.javaObjectType,
            Placeholder.shape(Shape.make(-1, 784))
        )

        val labels =
            tf.placeholder(Float::class.javaObjectType)


        val weights: Variable<Float> =
            tf.variable(Shape.make(784, 10), Float::class.javaObjectType)
        val weightsInit =
            tf.assign(weights, tf.zeros(constArray(tf, 784, 10), Float::class.javaObjectType))

        val biases: Variable<Float> =
            tf.variable(Shape.make(10), Float::class.javaObjectType)
        val biasesInit = tf.assign(biases, tf.zeros(constArray(tf, 10), Float::class.javaObjectType))

        // Build the graph
        // Build the graph
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

        val gradients = tf.gradients(crossEntropy, Arrays.asList(weights, biases))
        val alpha = tf.constant(0.2f)
        val weightGradientDescent =
            tf.train.applyGradientDescent(weights, alpha, gradients.dy<Float>(0))
        val biasGradientDescent =
            tf.train.applyGradientDescent(biases, alpha, gradients.dy<Float>(1))

        val predicted: Operand<Long> = tf.math.argMax(softmax, tf.constant(1))
        val expected: Operand<Long> = tf.math.argMax(labels, tf.constant(1))
        val accuracy/*: Operand<java.lang.Float>*/ = tf.math.mean(
            tf.dtypes.cast(
                tf.math.equal<Long>(predicted, expected),
                Float::class.javaObjectType
            ), constArray(tf, 0)
        )

        Session(graph).use { session ->
            // Initialize graph variables
            session.runner()
                .addTarget(weightsInit)
                .addTarget(biasesInit)
                .run()
            // Train the graph
            val batchIter: ImageDataset.ImageBatchIterator = dataset.trainingBatchIterator(TRAINING_BATCH_SIZE)
            while (batchIter.hasNext()) {
                val batch: ImageBatch = batchIter.next()
                Tensor.create(batch.shape(784), batch.images()).use({ batchImages ->
                    Tensor.create(batch.shape(10), batch.labels()).use({ batchLabels ->
                        session.runner()
                            .addTarget(weightGradientDescent)
                            .addTarget(biasGradientDescent)
                            .feed(images.asOutput(), batchImages)
                            .feed(labels.asOutput(), batchLabels)
                            .run()
                    })
                })
            }

            // Test the graph
            val testBatch: ImageBatch = dataset.testBatch()
            Tensor.create(testBatch.shape(784), testBatch.images()).use({ testImages ->
                Tensor.create(testBatch.shape(10), testBatch.labels()).use({ testLabels ->
                    session.runner()
                        .fetch(accuracy)
                        .feed(images.asOutput(), testImages)
                        .feed(labels.asOutput(), testLabels)
                        .run()[0].use { value -> println("Accuracy: " + value.floatValue()) }
                })
            })
        }
    }
}


