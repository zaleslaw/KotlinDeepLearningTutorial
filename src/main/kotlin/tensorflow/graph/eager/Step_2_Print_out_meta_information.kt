package tensorflow.graph.eager

import org.tensorflow.EagerSession
import org.tensorflow.op.Ops

/**
 * This example takes the code from the previous and
 * adds the printing of metadata about each operation.
 */
fun main() {
    EagerSession.create().use { session ->
        val tf = Ops.create(session)

        val aOps = tf.constant(10L)

        println("-------- aOps -----------------------")
        println("Name: ${aOps.op().name()}")
        println("Tensor type: ${aOps.op().type()}")
        println("Num outputs: ${aOps.op().numOutputs()}")

        val bOps = tf.constant(5L)

        println("-------- bOps -----------------------")
        println("Name: ${bOps.op().name()}")
        println("Tensor type: ${bOps.op().type()}")
        println("Num outputs: ${bOps.op().numOutputs()}")

        val addOps = tf.math.add(aOps, bOps)

        println("-------- addOps -----------------------")
        println("Name: ${addOps.op().name()}")
        println("Tensor type: ${addOps.op().type()}")
        println("Num outputs: ${addOps.op().numOutputs()}")

        println("Num dimensions in shape: ${addOps.z().shape().numDimensions()}")
        println("Num elements in tensor: ${addOps.z().tensor().numElements()}")

        println("Result: " + addOps.asOutput().tensor().longValue())
    }
}
