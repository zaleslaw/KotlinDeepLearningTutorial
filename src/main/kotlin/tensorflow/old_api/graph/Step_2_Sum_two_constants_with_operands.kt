package tensorflow.old_api.graph

import org.tensorflow.Graph
import org.tensorflow.Session
import org.tensorflow.op.Ops

/**
 * Defines the simplest Operand Graph: 10L + 5L via generated operands.
 */
fun main() {
    Graph().use { g ->
        Session(g).use { session ->
            val tf = Ops.create(g)
            val aOps = tf.constant(10L)
            val bOps = tf.constant(5L)
            val addOps = tf.math.add(aOps, bOps)

            println(session.runner().fetch(addOps).run()[0].longValue())
        }
    }
}