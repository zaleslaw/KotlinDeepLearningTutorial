package tensorflow.graph.static.graphbuilder

import org.apache.commons.lang3.ArrayUtils.toArray
import org.tensorflow.*
import java.util.*

/**
 * Builds the graph, adds gradients, print out it.
 */
fun main() {
    Graph().use { g ->
        Session(g).use { s ->
            val x1: Output<Float> = g.opBuilder("Placeholder", "x1")
                .setAttr("dtype", DataType.FLOAT)
                .build()
                .output<Float>(0)

            val x2: Output<Float> = g.opBuilder("Placeholder", "x2")
                .setAttr("dtype", DataType.FLOAT)
                .build()
                .output<Float>(0)

            val y0: Output<Float> = g.opBuilder("Square", "y0")
                .addInput(x1)
                .build()
                .output<Float>(0)

            val y1: Output<Float> = g.opBuilder("Square", "y1")
                .addInput(x1)
                .build()
                .output<Float>(0)

            val input = arrayOf(y0, x2)

            val y2: Output<Float> = g.opBuilder("AddN", "AddN").addInputList(input).build().output(0);
            val grads0: Array<Output<*>> = g.addGradients(y1, toArray(x1))

            println(grads0)
            println(grads0.size)
            println(grads0[0].dataType())

            val grads1: Array<Output<*>> = g.addGradients(y2, toArray(x1, x2))
            println(grads1)
            println(grads1.size)
            println(grads1[0].dataType())
            println(grads1[1].dataType())

            Tensors.create(3.0f).use { c1 ->
                Tensors.create(2.0f).use { c2 ->
                    AutoCloseableList(
                        s.runner()
                            .feed(x1, c1)
                            .feed(x2, c2)
                            .fetch(grads0[0])
                            .fetch(grads1[0])
                            .fetch(grads1[1])
                            .run()
                    ).use { outputs ->
                        println(outputs.size)
                        println(outputs[0].floatValue())
                        println(outputs[1].floatValue())
                        println(outputs[2].floatValue())
                    }
                }
            }
        }
    }
}

class AutoCloseableList<E : AutoCloseable?>(c: Collection<E>?) :
    ArrayList<E>(c), AutoCloseable {
    @Throws(Exception::class)
    override fun close() {
    }
}
