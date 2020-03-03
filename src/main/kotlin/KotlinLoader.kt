import org.tensorflow.SavedModelBundle
import org.tensorflow.Tensor
import org.tensorflow.TensorFlow
import java.util.*
import java.util.function.Function

fun main() {
    println(TensorFlow.version())

    val images = MnistUtils.mnistAsList(
        "src/main/resources/datasets/test/t10k-images-idx3-ubyte",
        "src/main/resources/datasets/test/t10k-labels-idx1-ubyte",
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

    exec(images, ::reshape)
}

private fun exec(images: MutableList<MnistUtils.MnistLabeledImage>, reshape: (DoubleArray) -> Tensor<*>?) {
    SavedModelBundle.load("src/main/resources/model1", "serve").use { bundle ->
        var counter = 0
        for (image in images) {
            val session = bundle.session()
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
    }
}