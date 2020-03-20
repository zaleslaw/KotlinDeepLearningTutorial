package dl4j.tensors

import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.ops.transforms.Transforms.*
import java.io.IOException
import java.util.*


/**
 * These are common functions that most python numpy users use for their daily work.
 * I've provided examples for all such users who are coming from the numpy environment to ND4J
 * You can view the cheatsheat and see the implementations and use cases here
 *
 * Following is the link to the cheatsheat I've implemented
 * https://www.dataquest.io/blog/images/cheat-sheets/numpy-cheat-sheet.pdf
 */
fun main() {
    /* A. IMPORTING/EXPORTING */
    // 1. np.loadtxt('file.txt') - From a text file
    /* A. IMPORTING/EXPORTING */ // 1. np.loadtxt('file.txt') - From a text file
    var readFromText: INDArray? = null
    try {
        readFromText = Nd4j.readNumpy(makeResourcePath("/numpy_cheatsheet/file.txt"))
        print("Read from text", readFromText)
    } catch (e: IOException) {
        e.printStackTrace()
    }

    // 2. np.genfromtxt('file.csv',delimiter=',') - From a CSV file
    var readFromCSV: INDArray? = null
    try {
        readFromCSV = Nd4j.readNumpy(makeResourcePath("/numpy_cheatsheet/file.csv"), ",")
        print("Read from csv", readFromCSV)
    } catch (e: IOException) {
        e.printStackTrace()
    }

    /* B. CREATING ARRAYS */
    // 1. np.array([1,2,3]) - One dimensional array
    /* B. CREATING ARRAYS */ // 1. np.array([1,2,3]) - One dimensional array
    val oneDArray = Nd4j.create(floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f), intArrayOf(6))
    print("One Dimensional Array", oneDArray)

    // 2. np.array([(1,2,3),(4,5,6)]) - Two dimensional array
    val twoDArray = Nd4j.create(floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f), intArrayOf(2, 3))
    print("Two Dimensional Array", twoDArray)

    // 3. np.zeros(3) - 1D array of length 3 all values 0
    val oneDZeros = Nd4j.zeros(3)
    print("One dimensional zeros", oneDZeros)

    // 4. np.ones((3,4)) - 3x4 array with all values 1
    val threeByFourOnes = Nd4j.ones(3, 4)
    print("3x4 ones", threeByFourOnes)

    // 5. np.eye(5) - 5x5 array of 0 with 1 on diagonal (Identity matrix)
    val fiveByFiveIdentity = Nd4j.eye(5)
    print("5x5 Identity", fiveByFiveIdentity)

    // 6. np.linspace(0,100,6) - Array of 6 evenly divided values from 0 to 100
    val zeroToHundredLinspaceOfSix = Nd4j.linspace(0, 100, 6)
    print("Zero to Hundred With linspace interval 6", zeroToHundredLinspaceOfSix)

    // 8. np.full((2,3),8) - 2x3 array with all values 8
    val allEights = Nd4j.valueArrayOf(intArrayOf(2, 3), 8.0)
    print("2x3 Eights", allEights)

    // 9. np.random.rand(4,5) - 4x5 array of random floats between 0-1
    val fourByFiveRandomZeroToOne = Nd4j.rand(*intArrayOf(4, 5))
    print("4x5 Random between zero and one", fourByFiveRandomZeroToOne)

    // 10. np.random.rand(6,7)*100 - 6x7 array of random floats between 0-100
    val sixBySevenRandomZeroToHundred = Nd4j.rand(*intArrayOf(6, 7)).mul(100)
    print("6x7 Random between zero and hundred", sixBySevenRandomZeroToHundred)


    /* C. INSPECTING PROPERTIES */
    // 1. arr.size - Returns number of elements in arr
    /* C. INSPECTING PROPERTIES */ // 1. arr.size - Returns number of elements in arr
    val size = fourByFiveRandomZeroToOne.length().toInt()
    println("Array size: $size")

    // 2. arr.shape - Returns dimensions of arr (rows, columns)
    val shape = fourByFiveRandomZeroToOne.shape()
    System.out.println("Array shape: " + Arrays.toString(shape))

    // 5. arr.tolist() - Convert arr to a Python list
    val bytes = fourByFiveRandomZeroToOne.data().asBytes()
    System.out.println("Array byte: " + Arrays.toString(bytes))
    val doubles = fourByFiveRandomZeroToOne.data().asDouble()
    System.out.println("Array doubles: " + Arrays.toString(doubles))
    val floats = fourByFiveRandomZeroToOne.data().asFloat()
    System.out.println("Array floats: " + Arrays.toString(floats))
    val ints = fourByFiveRandomZeroToOne.data().asInt()
    System.out.println("Array ints: " + Arrays.toString(ints)) //and so on...

    /* D. COPYING/SORTING/RESHAPING */
    // 1. np.copy(arr) - Copies arr to new memory
    /* D. COPYING/SORTING/RESHAPING */ // 1. np.copy(arr) - Copies arr to new memory
    val copy = fourByFiveRandomZeroToOne.dup()
    print("Copied array: ", copy)

    // 3. arr.sort() - Sorts arr
    val sortedArray = Nd4j.sort(fourByFiveRandomZeroToOne, true)
    print("Ascended sorted array: ", sortedArray)

    // 4. arr.sort(axis=0) - Sorts specific axis of arr
    val axisSortedArray = Nd4j.sort(fourByFiveRandomZeroToOne, 0, true)
    print("Ascended sorted array on zero axis: ", axisSortedArray)

    // 5. two_d_arr.flatten() - Flattens 2D array two_d_arr to 1D
    val flattened = Nd4j.toFlattened(fourByFiveRandomZeroToOne)
    print("Flattened array", flattened)

    // 6. arr.T - Transposes arr (rows become columns and vice versa)
    val transpose = fourByFiveRandomZeroToOne.transpose()
    print("Transposed array", transpose)

    // 7. arr.reshape(5,4) - Reshapes arr to 5 rows, 4 columns without changing data
    val reshaped = fourByFiveRandomZeroToOne.reshape(5, 4)
    print("Reshaped array", reshaped)


    /* F. COMBINING/SPLITTING */
    // 1. np.concatenate((arr1,arr2),axis=0) - Adds arr2 as rows to the end of arr1
    /* F. COMBINING/SPLITTING */ // 1. np.concatenate((arr1,arr2),axis=0) - Adds arr2 as rows to the end of arr1
    val concatenatedAxisZero = Nd4j.concat(0, Nd4j.create(3, 2), Nd4j.create(5, 2))
    print("Concatenated arrays on dimension zero", concatenatedAxisZero)

    // 2. np.concatenate((arr1,arr2),axis=1) - Adds arr2 as columns to end of arr1
    val concatenatedAxisOne = Nd4j.concat(1, Nd4j.create(3, 2), Nd4j.create(3, 5))
    print("Concatenated arrays on dimension 1", concatenatedAxisOne)


    /* G. INDEXING/SLICING/SUBSETTING */
    // 1. arr[5] - Returns the element at index 5
    /* G. INDEXING/SLICING/SUBSETTING */

    // 1. arr[5] - Returns the element at index 5
    /*val oneValue = fourByFiveRandomZeroToOne.getDouble(5)
    println("Get one value from 1D array: $oneValue")*/

    // 2. arr[2,4] - Returns the 2D array element on index [2][5]
    val oneValue2D = fourByFiveRandomZeroToOne.getDouble(2, 4)
    println("Get one value from 2D array: $oneValue2D")

    // 3. arr[1]=4 - Assigns array element on index 1 the value 4
    fourByFiveRandomZeroToOne.putScalar(1, 4)
    print("Assigned value to array (1 => 4)", fourByFiveRandomZeroToOne)

    // 4. arr[1,3]=10 - Assigns array element on index [1][3] the value 10
    fourByFiveRandomZeroToOne.putScalar(intArrayOf(1, 3), 10)
    print("Assigned value to array (1x3 => 10)", fourByFiveRandomZeroToOne)

    // 5. arr[0:3] - Returns the elements at indices 0,1,2 (On a 2D array: returns rows 0,1,2)
    val threeValuesArray = fourByFiveRandomZeroToOne[NDArrayIndex.interval(0, 3)]
    print("Get interval from array ([0:3])", threeValuesArray)

    // 6. arr[0:3,4] - Returns the elements on rows 0,1,2 at column 4
    val threeValuesArrayColumnFour =
        fourByFiveRandomZeroToOne[NDArrayIndex.interval(0, 3), NDArrayIndex.point(4)]
    print("Get interval from array ([0:3,4])", threeValuesArrayColumnFour)

    // 7. arr[:2] - Returns the elements at indices 0,1 (On a 2D array: returns rows 0,1)
    val threeValuesArrayAgain = fourByFiveRandomZeroToOne[NDArrayIndex.interval(0, 2)]
    print("Get interval from array ([:2])", threeValuesArrayAgain)

    // 8. arr[:,1] - Returns the elements at index 1 on all rows
    val allRowsIndexOne = fourByFiveRandomZeroToOne[NDArrayIndex.all(), NDArrayIndex.point(1)]
    print("Get interval from array ([:,1])", allRowsIndexOne)


    /* H. SCALAR MATH */
    // 1. np.add(arr,1) - Add 1 to each array element
    /* H. SCALAR MATH */ // 1. np.add(arr,1) - Add 1 to each array element
    val addOne = fourByFiveRandomZeroToOne.add(1)
    print("Add 1 to array", addOne)

    // 2. np.subtract(arr,2) - Subtract 2 from each array element
    val subtractTwo = fourByFiveRandomZeroToOne.sub(2)
    print("Subtract 2 from array", subtractTwo)

    // 3. np.multiply(arr,3) - Multiply each array element by 3
    val multiplyThree = fourByFiveRandomZeroToOne.mul(3)
    print("Multiply 3 to array", multiplyThree)

    // 4. np.divide(arr,4) - Divide each array element by 4 (returns np.nan for division by zero)
    val divideFour = fourByFiveRandomZeroToOne.div(4)
    print("Divide array by 4", divideFour)

    // 5. np.power(arr,5) - Raise each array element to the 5th power
    val pow: INDArray = pow(fourByFiveRandomZeroToOne, 5)
    print("5th power of array", pow)

    /* I. VECTOR MATH */
    // 1. np.add(arr1,arr2) - Elementwise add arr2 to arr1
    /* I. VECTOR MATH */ // 1. np.add(arr1,arr2) - Elementwise add arr2 to arr1
    val secondArray = Nd4j.create(*intArrayOf(4, 5)).add(10)
    val vectorAdd = fourByFiveRandomZeroToOne.add(secondArray)
    print("Vector add", vectorAdd)

    // 2. np.subtract(arr1,arr2) - Elementwise subtract arr2 from arr1
    val vectorSubtract = fourByFiveRandomZeroToOne.sub(secondArray)
    print("Vector subtract", vectorSubtract)

    // 3. np.multiply(arr1,arr2) - Elementwise multiply arr1 by arr2
    val vectorMultiply = fourByFiveRandomZeroToOne.mul(secondArray)
    print("Vector multiply", vectorMultiply)

    // 4. np.divide(arr1,arr2) - Elementwise divide arr1 by arr2
    val vectorDivide = fourByFiveRandomZeroToOne.div(secondArray)
    print("Vector divide", vectorDivide)

    // 5. np.power(arr1,arr2) - Elementwise raise arr1 raised to the power of arr2
    val power: INDArray = pow(fourByFiveRandomZeroToOne, secondArray)
    print("Vector power", power)

    // 7. np.sqrt(arr) - Square root of each element in the array
    val sqrt: INDArray = sqrt(fourByFiveRandomZeroToOne)
    print("Vector square root", sqrt)

    // 8. np.sin(arr) - Sine of each element in the array
    val sin: INDArray = sin(fourByFiveRandomZeroToOne)
    print("Vector sin", sin)

    // 9. np.log(arr) - Natural log of each element in the array
    val log: INDArray = log(fourByFiveRandomZeroToOne)
    print("Vector log", log)

    // 10. np.abs(arr) - Absolute value of each element in the array
    val abs: INDArray = abs(fourByFiveRandomZeroToOne)
    print("Vector abs", abs)

    // 11. np.ceil(arr) - Rounds up to the nearest int
    val ceil: INDArray = ceil(fourByFiveRandomZeroToOne)
    print("Vector ceil", ceil)

    // 12. np.floor(arr) - Rounds down to the nearest int
    val floor: INDArray = floor(fourByFiveRandomZeroToOne)
    print("Vector floor", floor)

    // 13. np.round(arr) - Rounds to the nearest int
    val round: INDArray = round(fourByFiveRandomZeroToOne)
    print("Vector round", round)

    /* J. STATISTICS */
    // 1. np.mean(arr,axis=0) - Returns mean along specific axis
    /* J. STATISTICS */ // 1. np.mean(arr,axis=0) - Returns mean along specific axis
    val mean = Nd4j.mean(fourByFiveRandomZeroToOne, 0)
    print("Mean on dimension zero", mean)

    // 2. arr.sum() - Returns sum of arr
    val sum = fourByFiveRandomZeroToOne.sumNumber()
    println("Sum: $sum")

    // 3. arr.min() - Returns minimum value of arr
    val min = fourByFiveRandomZeroToOne.minNumber()
    println("Min: $min")

    // 4. arr.max(axis=0) - Returns maximum value of specific axis
    val max = fourByFiveRandomZeroToOne.maxNumber()
    println("Max: $max")

    // 5. np.var(arr) - Returns the variance of array
    val `var` = Nd4j.`var`(fourByFiveRandomZeroToOne)
    print("Variance", `var`)

    // 6. np.std(arr,axis=1) - Returns the standard deviation of specific axis
    val std = Nd4j.std(fourByFiveRandomZeroToOne, 1)
    print("Standard deviation", std)

    // 7. arr.corrcoef() - Returns correlation coefficient of array
    //todo: Returns correlation coefficient of array
}

private fun print(tag: String, arr: INDArray) {
    println("----------------")
    println("$tag:\n$arr")
    println("----------------")
}

private fun print(tag: String, arrays: Array<INDArray>) {
    println("----------------")
    println(tag)
    for (array in arrays) {
        println("\n" + array)
    }
    println("----------------")
}

private fun makeResourcePath(template: String): String? {
    return object {}.javaClass.getResource(template).getPath()
}