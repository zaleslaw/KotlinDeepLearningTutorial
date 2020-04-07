package tensorflow.graph.static.graphbuilder

import org.tensorflow.DataType.INT64
import org.tensorflow.Graph
import org.tensorflow.Operation
import org.tensorflow.Session
import org.tensorflow.Tensors
import tensorflow.inference.printTFGraph

/**
 * Defines the simplest Graph: z = 10*x + 5*y via Graph builder.
 */
fun main() {
    Graph().use { g ->
        Session(g).use { session ->

            // Define constants

            val a: Operation = g.opBuilder("Const", "a")
                .setAttr("dtype", INT64)
                .setAttr("value", Tensors.create(10L))
                .build()

            val b: Operation = g.opBuilder("Const", "b")
                .setAttr("dtype", INT64)
                .setAttr("value", Tensors.create(5L))
                .build()

            // Define placeholders

            val x: Operation = g.opBuilder("Placeholder", "x")
                .setAttr("dtype", INT64)
                .build()

            val y: Operation = g.opBuilder("Placeholder", "y")
                .setAttr("dtype", INT64)
                .build()


            // Define functions

            val ax: Operation = g.opBuilder("Mul", "ax")
                .addInput(a.output<Long>(0))
                .addInput(x.output<Long>(0))
                .build()

            val by: Operation = g.opBuilder("Mul", "by")
                .addInput(b.output<Long>(0))
                .addInput(y.output<Long>(0))
                .build()

            val result: Operation = g.opBuilder("Add", "result")
                .addInput(ax.output<Long>(0))
                .addInput(by.output<Long>(0))
                .build()

            println("Result operation type ${result.type()}")

            printTFGraph(g)

            // Run the calculations

            val tensor = session
                .runner()
                .fetch("result")
                .feed("x", Tensors.create(1L))
                .feed("y", Tensors.create(2L))
                .run()[0]

            println(tensor.longValue())
        }
    }
}