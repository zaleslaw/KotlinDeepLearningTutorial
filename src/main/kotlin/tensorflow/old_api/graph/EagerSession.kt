package tensorflow.old_api.graph

import org.tensorflow.EagerSession
import org.tensorflow.op.Ops

fun main() {
    EagerSession.create().use { session ->
        val tf = Ops.create(session)
        val aOps = tf.constant(10L)
        val bOps = tf.constant(5L)
        val addOps = tf.math.add(aOps, bOps)

        println(addOps.asOutput().tensor().longValue())
    }
}

