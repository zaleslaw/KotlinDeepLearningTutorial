import jdk.internal.util.xml.impl.Input
import org.tensorflow.DataType
import org.tensorflow.EagerSession
import org.tensorflow.Shape
import org.tensorflow.TensorFlow
import org.tensorflow.op.Ops

fun main() {
    println(TensorFlow.version())

    EagerSession.create().use { session ->
        val tf = Ops.create(session)
        val aOps = tf.constant(10L)
        val bOps = tf.constant(5L)
        val addOps = tf.math.add(aOps, bOps)

        println(addOps.asOutput().tensor().longValue())
    }
}

