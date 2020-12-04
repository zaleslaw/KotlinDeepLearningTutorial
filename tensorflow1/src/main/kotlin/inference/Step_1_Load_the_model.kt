package inference

import org.tensorflow.SavedModelBundle
import org.tensorflow.Session
import org.tensorflow.Tensor
import util.MnistUtils
import java.util.*

const val IMAGE_PATH = "src/main/resources/datasets/test/t10k-images-idx3-ubyte"
const val LABEL_PATH = "src/main/resources/datasets/test/t10k-labels-idx1-ubyte"
const val PATH_TO_MODEL = "src/main/resources/model1"

fun main() {
    val images = MnistUtils.mnistAsList(
        IMAGE_PATH,
        LABEL_PATH, Random(0), 10000
    )
    println("Amount of images: " + images.size)

    SavedModelBundle.load(PATH_TO_MODEL, "serve").use { bundle ->
        val session = bundle.session()

        evaluateTFModel(images, session)
        session.close()
    }
}

private fun reshape(doubles: DoubleArray): Tensor<*>? {
    val reshaped = Array(
        1
    ) { Array(28) { FloatArray(28) } }
    for (i in doubles.indices) reshaped[0][i / 28][i % 28] = doubles[i].toFloat()
    return Tensor.create(reshaped)
}

private fun evaluateTFModel(
    images: MutableList<MnistUtils.MnistLabeledImage>,
    session: Session
) {
    var counter = 0
    for (image in images) {
        val result = predictOnImage(session, ::reshape, image)
        if (result[0].toInt() == image.label)
            counter++
    }

    println("Accuracy is : " + (counter.toDouble() / images.size))
}

private fun predictOnImage(
    session: Session,
    reshape: (DoubleArray) -> Tensor<*>?,
    image: MnistUtils.MnistLabeledImage
): LongArray {
    val runner = session.runner()
    return runner.feed("Placeholder", reshape(image.pixels))
        .fetch("ArgMax")
        .run()[0]
        .copyTo(LongArray(1))
}
