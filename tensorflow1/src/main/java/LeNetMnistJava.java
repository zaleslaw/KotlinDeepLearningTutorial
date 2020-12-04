import org.tensorflow.Shape;
import org.tensorflow.*;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.*;
import org.tensorflow.op.math.Add;
import org.tensorflow.op.math.Mean;
import org.tensorflow.op.nn.*;
import org.tensorflow.op.random.TruncatedNormal;
import org.tensorflow.op.train.ApplyGradientDescent;
import training.util.ImageBatch;
import training.util.ImageDataset;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class LeNetMnistJava {
    // Hyper-parameters
    private static final float LEARNING_RATE = 0.2f;
    private static final int EPOCHS = 10;
    private static final int TRAINING_BATCH_SIZE = 500;

    // Image pre-processing constants
    private static final long NUM_LABELS = 10L;
    private static final long NUM_CHANNELS = 1L;
    private static final long IMAGE_SIZE = 28L;

    private static final int VALIDATION_SIZE = 0;
    private static final long SEED = 12L;
    private static final String PADDING_TYPE = "SAME";

    // Tensor names
    private static final String INPUT_NAME = "input";
    private static final String OUTPUT_NAME = "output";
    private static final String TRAINING_LOSS = "training_loss";

    public static void main(String[] args) {
        ImageDataset dataset = ImageDataset.Companion.create(VALIDATION_SIZE);

        try (Graph graph = new Graph()) {
            Ops tf = Ops.create(graph);

            // Define placeholders
            Placeholder<Float> images = tf.withName(INPUT_NAME).placeholder(Float.class,
                    Placeholder.shape(Shape.make(-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)));

            Placeholder<Float> labels = tf.placeholder(Float.class);

            // First conv layer

            // Generate random data to fill the weight matrix
            TruncatedNormal<Float> truncatedNormal = tf.random.truncatedNormal(
                    tf.constant(new long[]{5, 5, NUM_CHANNELS, 32}),
                    Float.class,
                    TruncatedNormal.seed(SEED)
            );

            Variable<Float> conv1Weights = tf.variable(Shape.make(5L, 5L, NUM_CHANNELS, 32), Float.class);

            Assign<Float> conv1WeightsInit = tf.assign(conv1Weights, tf.math.mul(truncatedNormal, tf.constant(0.1f)));

            Conv2d<Float> conv1 = tf.nn.conv2d(images, conv1Weights, longListOf(1L, 1L, 1L, 1L), PADDING_TYPE);

            Variable<Float> conv1Biases = tf.variable(Shape.make(32), Float.class);

            Assign<Float> conv1BiasesInit = tf.assign(conv1Biases, tf.zeros(constArray(tf, 32), Float.class));

            Relu<Float> relu1 = tf.nn.relu(tf.nn.biasAdd(conv1, conv1Biases));

            // First pooling layer
            MaxPool<Float> pool1 = tf.nn.maxPool(relu1,
                    tf.constant(new int[]{1, 2, 2, 1}), tf.constant(new int[]{1, 2, 2, 1}), PADDING_TYPE);

            // Second conv layer
            TruncatedNormal<Float> truncatedNormal2 = tf.random.truncatedNormal(tf.constant(new long[]{5, 5, 32, 64}),
                    Float.class, TruncatedNormal.seed(SEED));

            Variable<Float> conv2Weights = tf.variable(Shape.make(5, 5, 32, 64), Float.class);

            Assign<Float> conv2WeightsInit = tf.assign(conv2Weights, tf.math.mul(truncatedNormal2, tf.constant(0.1f)));

            Conv2d<Float> conv2 = tf.nn.conv2d(pool1, conv2Weights, longListOf(1L, 1L, 1L, 1L), PADDING_TYPE);

            Variable<Float> conv2Biases = tf.variable(Shape.make(64), Float.class);

            Assign<Float> conv2BiasesInit = tf.assign(conv2Biases, tf.zeros(constArray(tf, 64), Float.class));

            Relu<Float> relu2 = tf.nn.relu(tf.nn.biasAdd(conv2, conv2Biases));

            // Second pooling layer
            MaxPool<Float> pool2 = tf.nn.maxPool(relu2, tf.constant(new int[]{1, 2, 2, 1}), tf.constant(new int[]{1, 2, 2, 1}), PADDING_TYPE);

            // Flatten inputs
            Slice<Integer> slice = tf.slice(tf.shape(pool2), tf.constant(new int[]{0}), tf.constant(new int[]{1}));

            List<Operand<Integer>> mutableListOf = listOfIntOperands(slice, tf.constant(new int[]{-1}));

            Reshape<Float> flatten = tf.reshape(pool2, tf.concat(mutableListOf, tf.constant(0)));

            // Fully connected layer
            TruncatedNormal<Float> truncatedNormal3 = tf.random.truncatedNormal(tf.constant(new long[]{IMAGE_SIZE * IMAGE_SIZE * 4, 512}), Float.class, TruncatedNormal.seed(SEED));

            Variable<Float> fc1Weights = tf.variable(Shape.make(IMAGE_SIZE * IMAGE_SIZE * 4, 512), Float.class);

            Assign<Float> fc1WeightsInit = tf.assign(fc1Weights, tf.math.mul(truncatedNormal3, tf.constant(0.1f)));

            Variable<Float> fc1Biases = tf.variable(Shape.make(512), Float.class);

            Assign<Float> fc1BiasesInit = tf.assign(fc1Biases, tf.fill(tf.constant(new int[]{512}), tf.constant(0.1f)));

            Relu<Float> relu3 = tf.nn.relu(tf.math.add(tf.linalg.matMul(flatten, fc1Weights), fc1Biases));

            // Softmax layer
            TruncatedNormal<Float> truncatedNormal4 = tf.random.truncatedNormal(tf.constant(new long[]{512, NUM_LABELS}), Float.class, TruncatedNormal.seed(SEED));

            Variable<Float> fc2Weights = tf.variable(Shape.make(512, NUM_LABELS), Float.class);

            Assign<Float> fc2WeightsInit = tf.assign(fc2Weights, tf.math.mul(truncatedNormal4, tf.constant(0.1f)));

            Variable<Float> fc2Biases = tf.variable(Shape.make(NUM_LABELS), Float.class);

            Assign<Float> fc2BiasesInit = tf.assign(fc2Biases, tf.fill(tf.constant(new int[]{(int) NUM_LABELS}), tf.constant(0.1f)));

            Add<Float> logits = tf.math.add(tf.linalg.matMul(relu3, fc2Weights), fc2Biases);

            // Predicted outputs
            Softmax<Float> prediction = tf.withName(OUTPUT_NAME).nn.softmax(logits);

            SoftmaxCrossEntropyWithLogits<Float> batchLoss = tf.nn.softmaxCrossEntropyWithLogits(logits, labels);

            Mean<Float> loss = tf.withName(TRAINING_LOSS).math.mean(batchLoss.loss(), tf.constant(0));

            // Define gradients
            Constant<Float> learningRate = tf.constant(LEARNING_RATE);

            List<Variable<Float>> variables = listOfVariables(conv1Weights, conv1Biases, conv2Weights, conv2Biases, fc1Weights, fc1Biases, fc2Weights, fc2Biases);

            Gradients gradients = tf.gradients(loss, variables);

            // Set up the SGD for all variables
            ArrayList<ApplyGradientDescent<Float>> variablesGD = new ArrayList<>();

            for (int i = 0; i < variables.size(); i++) {
                variablesGD.add(i, tf.train.applyGradientDescent(variables.get(i), learningRate, gradients.dy(i)));
            }

            List<Assign<Float>> variablesInit = listOfAssigns(conv1WeightsInit, conv1BiasesInit, conv2WeightsInit, conv2BiasesInit, fc1WeightsInit, fc1BiasesInit, fc2WeightsInit, fc2BiasesInit);

            try (Session session = new Session(graph)) {
                // Initialize graph variables
                Session.Runner runner = session.runner();
                variablesInit.forEach(runner::addTarget);
                runner.run();

                // Train the graph
                for (int i = 0; i < EPOCHS; i++) {
                    ImageDataset.ImageBatchIterator batchIter = dataset.trainingBatchIterator(TRAINING_BATCH_SIZE);

                    while (batchIter.hasNext()) {
                        ImageBatch trainBatch = batchIter.next();

                        Tensor<?> batchImages = Tensor.create(new long[]{trainBatch.size(), IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS}, trainBatch.images());
                        Tensor<?> batchLabels = Tensor.create(new long[]{trainBatch.size(), 10}, trainBatch.labels());

                        Session.Runner trainingRunner = session.runner(); // get new Runner
                        variablesGD.forEach(trainingRunner::addTarget);

                        float lossValue = trainingRunner
                                .feed(images.asOutput(), batchImages)
                                .feed(labels.asOutput(), batchLabels)
                                .fetch(TRAINING_LOSS)
                                .run().get(0).floatValue();

                        System.out.println("epochs " + i + " lossValue " + lossValue);

                    }
                }

                // Evaluate the model
                Operand<Long> predicted = tf.math.argMax(prediction, tf.constant(1));
                Operand<Long> expected = tf.math.argMax(labels, tf.constant(1));

                // Define multi-classification metric
                Mean<Float> accuracy = tf.math.mean(
                        tf.dtypes.cast(
                                tf.math.equal(predicted, expected),
                                Float.class
                        ), constArray(tf, 0)
                );

                ImageBatch testBatch = dataset.testBatch();

                Tensor<?> testImages = Tensor.create(new long[]{testBatch.size(), IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS}, testBatch.images());
                Tensor<?> testLabels = Tensor.create(new long[]{testBatch.size(), 10}, testBatch.labels());

                float accuracyValue = session.runner()
                        .fetch(accuracy)
                        .feed(images.asOutput(), testImages)
                        .feed(labels.asOutput(), testLabels)
                        .run().get(0).floatValue();

                System.out.println("Accuracy: " + accuracyValue);

            }
        }
    }


    @SafeVarargs
    private static List<Assign<Float>> listOfAssigns(Assign<Float>... assigns) {
        List<Assign<Float>> res = new ArrayList<>();
        Collections.addAll(res, assigns);
        return res;
    }

    @SafeVarargs
    private static List<Variable<Float>> listOfVariables(Variable<Float>... variables) {
        List<Variable<Float>> res = new ArrayList<>();
        Collections.addAll(res, variables);
        return res;
    }


    @SafeVarargs
    private static List<Operand<Integer>> listOfIntOperands(Operand<Integer>... operands) {
        List<Operand<Integer>> res = new ArrayList<>();
        Collections.addAll(res, operands);
        return res;
    }

    private static List<Long> longListOf(long... elements) {
        List<Long> res = new ArrayList<>();
        for (long element : elements) {
            res.add(element);
        }
        return res;
    }

    public static Operand<Integer> constArray(Ops tf, int... i) {
        return tf.constant(i);
    }
}
