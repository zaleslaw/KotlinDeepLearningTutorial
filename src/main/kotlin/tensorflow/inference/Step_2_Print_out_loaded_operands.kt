package tensorflow.inference

import org.tensorflow.Graph
import org.tensorflow.GraphOperation
import org.tensorflow.SavedModelBundle

fun main() {
    SavedModelBundle.load(PATH_TO_MODEL, "serve").use { bundle ->
        val session = bundle.session()
        val graph = bundle.graph()

        printTFGraph(graph)

        session.close()
    }
}

fun printTFGraph(graph: Graph) {
    val operations = graph.operations()

    while (operations.hasNext()) {
        val operation = operations.next() as GraphOperation
        println("Name: " + operation.name() + "; Type: " + operation.type() + "; Out #tensors:  " + operation.numOutputs())
    }
}