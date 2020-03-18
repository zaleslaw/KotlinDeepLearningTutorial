package tensorflow.old_api.graph

import org.tensorflow.Tensor

/**
 * Load data to Tensor and copy from the Tensor to primitive matrix.
 */
fun main() {
    // Rank 2 Tensor
    // 3x2 matrix of floats.
    val matrix = Array(3) { FloatArray(2) }
    matrix[1][0] = 42.0f

    val rank2Tensor = Tensor.create(matrix, Float::class.javaObjectType)
    println("DataType: " + rank2Tensor.dataType().name)
    println("NumElements: " + rank2Tensor.numElements())
    println("NumDimensions: " + rank2Tensor.numDimensions())

    Tensor.create(matrix).use { t ->
        val copy = Array(3) { FloatArray(2) }
        println("Copied element: " + t.copyTo(copy)[1][0])
    }
}

