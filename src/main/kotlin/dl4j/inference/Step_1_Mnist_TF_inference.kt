package dl4j.inference

import org.apache.commons.io.IOUtils
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.tensorflow.conversion.graphrunner.GraphRunner;
import java.io.File
import java.io.FileInputStream
import java.util.*

const val PATH_TO_MODEL = "src/main/resources/model1/saved_model.pb"

/**
 * Failed with the next stacktrace
 * Exception in thread "main" java.lang.IllegalArgumentException: Unable to parse protobuf
at org.nd4j.tensorflow.conversion.graphrunner.GraphRunner.<init>(GraphRunner.java:175)
at org.nd4j.tensorflow.conversion.graphrunner.GraphRunner$GraphRunnerBuilder.build(GraphRunner.java:106)
at dl4j.inference.Step_1_Mnist_TF_inferenceKt.main(Step_1_Mnist_TF_inference.kt:22)
at dl4j.inference.Step_1_Mnist_TF_inferenceKt.main(Step_1_Mnist_TF_inference.kt)
Caused by: java.lang.IllegalStateException: ERROR: Unable to import graph Invalid GraphDef
at org.nd4j.tensorflow.conversion.TensorflowConversion.loadGraph(TensorflowConversion.java:338)
at org.nd4j.tensorflow.conversion.graphrunner.GraphRunner.<init>(GraphRunner.java:144)
... 3 more
 */
fun main() {
    val inputs: List<String> = Arrays.asList("flatten_2_input")

    //load the graph from the classpath
    val content: ByteArray = IOUtils.toByteArray(FileInputStream(File(PATH_TO_MODEL)))
    val dataSetIterator: DataSetIterator = MnistDataSetIterator(1, 1)
    val predict = dataSetIterator.next().features

    //run the graph using nd4j
    GraphRunner.builder().graphBytes(content).inputNames(inputs).build().use { graphRunner ->
        val inputMap: MutableMap<String, INDArray> = HashMap()
        inputMap[inputs[0]] = predict
        val run: Map<String, INDArray> = graphRunner.run(inputMap)
        println("Run result $run")
    }
}