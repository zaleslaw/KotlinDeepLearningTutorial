package tensorflow.inference

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
