package tensorflow.graph.static.graphbuilder

import org.tensorflow.Graph
import org.tensorflow.Output
import org.tensorflow.Session
import org.tensorflow.Tensors
import tensorflow.inference.printTFGraph

/**
 * Defines the simplest Graph: 10L + 5L via Graph builder.
 */
fun main() {
    Graph().use { g ->
        Session(g).use { session ->
            Tensors.create(10L).use { c1 ->
                Tensors.create(5L).use { c2 ->
                    println("---Empty graph---")
                    printTFGraph(g)

                    val aOps = g
                        .opBuilder("Const", "aOps")
                        .setAttr("dtype", c1.dataType())
                        .setAttr("value", c1)
                        .build()
                        .output<Long>(0)

                    println("---Graph with added aOps operand---")
                    printTFGraph(g)

                    val bOps = g
                        .opBuilder("Const", "bOps")
                        .setAttr("dtype", c1.dataType())
                        .setAttr("value", c2)
                        .build()
                        .output<Long>(0)

                    println("---Graph with added bOps operand---")
                    printTFGraph(g)

                    val graph = g
                        .opBuilder("Add", "Add")
                        .addInput(aOps)
                        .addInput(bOps)
                        .build()

                    println("---Final graph---")
                    printTFGraph(g)

                    val addOps: Output<Long> = graph
                        .output(0);

                    println(session.runner().fetch(addOps).run()[0].longValue())
                }
            }
        }
    }
}