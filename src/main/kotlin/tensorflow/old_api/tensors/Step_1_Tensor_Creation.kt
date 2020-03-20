package tensorflow.old_api.tensors

import org.tensorflow.Tensor

/**
 * Creates a few tensors of ranks 0, 1, 2
 */
fun main() {
    // Rank 0 Tensor
    val rank0Tensor = Tensor.create(42L, Long::class.javaObjectType)
    println("DataType: " + rank0Tensor.dataType().name)
    println("NumElements: " + rank0Tensor.numElements())
    println("NumDimensions: " + rank0Tensor.numDimensions())
    println("Value: " + rank0Tensor.longValue())

    // Rank 1 Tensor
    val vector = intArrayOf(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    val rank1Tensor = Tensor.create(vector, Int::class.javaObjectType)
    println("DataType: " + rank1Tensor.dataType().name)
    println("NumElements: " + rank1Tensor.numElements())
    println("NumDimensions: " + rank1Tensor.numDimensions())

    // Rank 2 Tensor
    // 3x2 matrix of floats.
    val matrix = Array(3) { FloatArray(2) }
    val rank2Tensor = Tensor.create(matrix, Float::class.javaObjectType)
    println("DataType: " + rank2Tensor.dataType().name)
    println("NumElements: " + rank2Tensor.numElements())
    println("NumDimensions: " + rank2Tensor.numDimensions())
}

