package tensorflow.old_api.inference

import org.tensorflow.*
import util.MnistUtils
import java.util.*

const val IMAGE_PATH = "src/resources/datasets/test/t10k-images-idx3-ubyte"
const val LABEL_PATH = "src/resources/datasets/test/t10k-labels-idx1-ubyte"

fun main() {
    val images = MnistUtils.mnistAsList(
        IMAGE_PATH,
        LABEL_PATH,
        Random(0),
        10000
    )

    fun reshape(doubles: DoubleArray): Tensor<*>? {
        val reshaped = Array(
            1
        ) { Array(28) { FloatArray(28) } }
        for (i in doubles.indices) reshaped[0][i / 28][i % 28] = doubles[i].toFloat()
        return Tensor.create(reshaped)
    }

    predictOnImagesWithTensor(images, ::reshape)
}

private fun predictOnImagesWithTensor(
    images: MutableList<MnistUtils.MnistLabeledImage>,
    reshape: (DoubleArray) -> Tensor<*>?
) {
    SavedModelBundle.load("src/resources/model1", "serve").use { bundle ->
        val session = bundle.session()

        val graph = bundle.graph()

        printTFGraph(graph)

        var counter = 0

        for (image in images) {
            val runner = session.runner()
            val result = runner.feed("Placeholder", reshape(image.pixels))
                .fetch("ArgMax")
                .run()[0]
                .copyTo(LongArray(1))
            if (result[0].toInt() == image.label)
                counter++

        }
        println(counter)
        println(images.size)

        session.close()
    }
}

private fun printTFGraph(graph: Graph) {
    val operations = graph.operations()

    while (operations.hasNext()) {
        val operation = operations.next() as GraphOperation
        println("Name: " + operation.name() + "; Type: " + operation.type() + "; Out #tensors:  " + operation.numOutputs())
        /*  for (i in 0 until operation.numOutputs()){
              println("       " + i + ":  " + operation.output<Any?>(i))
          }*/
    }
}

//private fun printTFGraphViaRunner(runner: Session.Runner) {
//    val tensors = runner.run()
//    for(tensor in tensors){
//        println("Data type: " + tensor.dataType() + " Dimensions:" + tensor.numDimensions() + " " + tensor.shape().toString())
//    }
//}