package tensorflow.old_api.inference

import org.tensorflow.Graph
import org.tensorflow.GraphOperation
import org.tensorflow.SavedModelBundle

/**
 * In this example the loaded graph was serialized and added to the initial graph with the special prefix "imported"
 */
fun main() {
    SavedModelBundle.load(PATH_TO_MODEL, "serve").use { bundle ->
        val session = bundle.session()
        val graph = bundle.graph()

        val serializedGraph = graph.toGraphDef()
        println(serializedGraph.size)

        graph.importGraphDef(serializedGraph, "imported")

        printTFGraph(graph)

        session.close()
    }
}

private fun printTFGraph(graph: Graph) {
    val operations = graph.operations()

    while (operations.hasNext()) {
        val operation = operations.next() as GraphOperation
        println("Name: " + operation.name() + "; Type: " + operation.type() + "; Out #tensors:  " + operation.numOutputs())
    }
}