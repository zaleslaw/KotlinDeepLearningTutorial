import jdk.internal.util.xml.impl.Input
import org.tensorflow.DataType
import org.tensorflow.EagerSession
import org.tensorflow.Shape
import org.tensorflow.TensorFlow
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Placeholder

fun main() {
    println(TensorFlow.version())

    EagerSession.create().use { session ->
        val tf = Ops.create(session)
        val options = null
        val x: Input = tf.placeholder(DataType.FLOAT, options)
        val weight: Input = tf.variable((784, 10))
        val b: Input = tf.variable(FloatArray(10))
        val y: Input = tf.math().add(tf.batchMatMulV2(x, weight), b)

        // Loss
        // Loss
        val y_: Input =
            tf.placeholder(DataType.FLOAT, Shape.make(-1, 10))
        val cross_entropy: Input =
            tf.math().mean(tf.nn().softmaxCrossEntropyWithLogits(y, y_).loss(), tf.constant(0))

//https://github.com/yinmazhong/tf-java/blob/fce7741be744b6de4e1c38f0bfa77b9e11238bb0/src/main/java/kbs/Main.java
        println(addOps.asOutput().tensor().longValue())
    }
}

