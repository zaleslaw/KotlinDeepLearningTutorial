package tensorflow.old_api.graph.static.graphbuilder

import org.tensorflow.Graph
import org.tensorflow.Output
import org.tensorflow.Session
import org.tensorflow.Tensors

/**
 * Defines the simplest Graph: 10L + 5L via Graph builder.
 */
fun main() {
    Graph().use { g ->
        Session(g).use { session ->
            Tensors.create(10L).use { c1 ->
                Tensors.create(5L).use { c2 ->
                    val aOps = g
                        .opBuilder("Const", "aOps")
                        .setAttr("dtype", c1.dataType())
                        .setAttr("value", c1)
                        .build()
                        .output<Long>(0)

                    val bOps = g
                        .opBuilder("Const", "bOps")
                        .setAttr("dtype", c1.dataType())
                        .setAttr("value", c2)
                        .build()
                        .output<Long>(0)


                    val addOps: Output<Long> = g
                        .opBuilder("Add", "Add")
                        .addInput(aOps)
                        .addInput(bOps)
                        .build()
                        .output(0);

                    println(session.runner().fetch(addOps).run()[0].longValue())
                }
            }
        }
    }
}