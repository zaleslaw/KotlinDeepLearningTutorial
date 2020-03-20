package dl4j.graph

import org.nd4j.autodiff.samediff.SameDiff
import org.nd4j.linalg.factory.Nd4j
import java.util.*

/**
 * SameDiff is an automatic differentiation library built on top of ND4J.
 * It can be considered comparable to TensorFlow or PyTorch,
 * in that users can define a set of tensor operations (a graph of operations, defining the "forward pass")
 * and SameDiff will automatically differentiate the graph.
 */
fun main() {
    // Define the graph
    val graph = SameDiff.create()

    // Define variables and operations
    val values = Nd4j.ones(3, 4)
    val variable = graph.`var`("myVariable", values)
    val plusOne = variable.add("add", 1.0) // Name: automatically generated as "add_scalar"
    val mulTen = variable.mul("mulTen", 10.0) // Name: Defined to be "mulTen"

    println("Graph summary")
    println(graph.summary())

    // Print out the graph parts
    val allVariables = graph.variables()
    println("Variables: $allVariables")
    for (`var` in allVariables) {
        val varShape = `var`.shape
        println(`var`.name() + " - shape " + Arrays.toString(varShape))
    }

    // Print out graph operations
    val ops = graph.ops()
    for (df in ops) {
        val inputsToFunction = df.args() // Inputs
        val outputsOfFunction = df.outputVariables() // Outputs

        println(
            "Op: " + df.opName() + ", inputs: " + Arrays.toString(inputsToFunction) + ", outputs: " +
                    Arrays.toString(outputsOfFunction)
        )
    }

    //Now, let's execute the graph forward pass:
    graph.output(emptyMap(), "mulTen")

    val variableArr = variable.arr //We can get arrays directly from the variables
    println("Initial variable values:\n$variableArr")

    //TODO: Error behaviour: Empty values
    val plusOneArr = graph.getArrForVarName("add")
    println("'plusOne' values:\n$plusOneArr")

    val mulTenArr = graph.getArrForVarName("mulTen")
    println("'mulTen' values:\n$mulTenArr")

}