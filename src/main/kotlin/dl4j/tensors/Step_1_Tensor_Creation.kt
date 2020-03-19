package dl4j.tensors

import org.nd4j.linalg.factory.Nd4j

fun main() {

    // Rank 1 Tensor
    val rank1Tensor = Nd4j.create(10)
    println("DataType: " + rank1Tensor.dataType().name)
    println("Shape: " + rank1Tensor.shape()[0])
    println("Rank: " + rank1Tensor.rank())
    println("Is a vector:        " + rank1Tensor.isVector);
    println("Is a scalar:        " + rank1Tensor.isScalar);
    println("Is a matrix:        " + rank1Tensor.isMatrix);
    println("Is a square matrix: " + rank1Tensor.isSquare);
    println(rank1Tensor)

    val updatedTensor = rank1Tensor.add(1).mul(2).putScalar(0, 100)
    println(updatedTensor)

    // Rank 2 Tensor
    val rank2Tensor = Nd4j.create(3, 2)
    println("DataType: " + rank2Tensor.dataType().name)
    println("Shape: " + rank2Tensor.shape()[0])
    println("Rank: " + rank2Tensor.rank())
    println("Is a vector:        " + rank2Tensor.isVector);
    println("Is a scalar:        " + rank2Tensor.isScalar);
    println("Is a matrix:        " + rank2Tensor.isMatrix);
    println("Is a square matrix: " + rank2Tensor.isSquare);

    val updatedMatrix = rank2Tensor.sub(10).div(100)
    println(updatedMatrix)

}