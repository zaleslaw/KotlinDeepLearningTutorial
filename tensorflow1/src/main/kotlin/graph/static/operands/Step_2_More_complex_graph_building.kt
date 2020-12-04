package graph.static.operands

import inference.printTFGraph
import org.tensorflow.Graph
import org.tensorflow.Session
import org.tensorflow.Tensors
import org.tensorflow.op.Ops

/**
 * Defines the simplest Operand Graph: z = 10*x + 5*y via generated operands.
 */
fun main() {
    Graph().use { g ->
        Session(g).use { session ->
            val tf = Ops.create(g)
            val a = tf.constant(10)
            val b = tf.constant(5)
            val x = tf.placeholder(Int::class.javaObjectType)
            val y = tf.placeholder(Int::class.javaObjectType)
            val ax = tf.math.mul(a, x)
            val by = tf.math.mul(b, y)
            val z = tf.math.add(ax, by)

            printTFGraph(g)

            // Run the calculations
            val tensor = session
                .runner()
                .fetch(z)
                .feed(x, Tensors.create(1))
                .feed(y, Tensors.create(2))
                .run()[0]

            println(tensor.intValue())
        }
    }
}
