package tensorflow.tensors

import org.tensorflow.Graph
import org.tensorflow.Session
import org.tensorflow.Tensor
import org.tensorflow.op.Ops

fun main() {
    val vector1 = intArrayOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    val vector2 = intArrayOf(1, 1, 1, 1, 1, 1, 1, 1, 1, 1)

    val t1 = Tensor.create(vector1)
    val t2 = Tensor.create(vector2)
    var t3: Tensor<*>

    Graph().use { g ->
        Session(g).use { session ->
            val tf = Ops.create(g)
            val aOps = tf.placeholder(Int::class.javaObjectType)
            val bOps = tf.placeholder(Int::class.javaObjectType)

            // t3 = t1 + t2
            val addOps = tf.math.add(aOps, bOps)

            t3 = session
                .runner()
                .feed(aOps.asOutput(), t1)
                .feed(bOps.asOutput(), t2)
                .fetch(addOps)
                .run()[0]

            printTensorData(t3)

            // t3 = t1 - t2
            val subOps = tf.math.sub(aOps, bOps)

            t3 = session
                .runner()
                .feed(aOps.asOutput(), t1)
                .feed(bOps.asOutput(), t2)
                .fetch(subOps)
                .run()[0]

            printTensorData(t3)

            // t3 = t1 * t2
            val mulOps = tf.math.mul(aOps, bOps)

            t3 = session
                .runner()
                .feed(aOps.asOutput(), t1)
                .feed(bOps.asOutput(), t2)
                .fetch(mulOps)
                .run()[0]

            printTensorData(t3)

            // t3 = t1 / t2
            val divOps = tf.math.div(aOps, bOps)

            t3 = session
                .runner()
                .feed(aOps.asOutput(), t1)
                .feed(bOps.asOutput(), t2)
                .fetch(divOps)
                .run()[0]

            printTensorData(t3)
        }
    }
}

private fun printTensorData(t3: Tensor<*>) {
    println("DataType: " + t3.dataType().name)
    println("NumElements: " + t3.numElements())
    println("NumDimensions: " + t3.numDimensions())

    t3.use { t ->
        val copy = intArrayOf(1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
        t.copyTo(copy)

        println(copy.contentToString())
    }
}
