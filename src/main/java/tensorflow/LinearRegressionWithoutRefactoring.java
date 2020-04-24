package tensorflow;

import com.sun.tools.javac.util.List;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Shape;
import org.tensorflow.Tensor;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.*;
import org.tensorflow.op.math.Add;
import org.tensorflow.op.math.Div;
import org.tensorflow.op.math.Mul;
import org.tensorflow.op.math.Pow;
import org.tensorflow.op.train.ApplyGradientDescent;

import java.util.Random;

class LinearRegressionWithoutRefactoring {
    /**
     * Amount of data points.
     */
    private static int n = 10;

    /**
     * This value is used to fill the Y placeholder in prediction.
     */
    private static float NO_MEANING_VALUE_TO_PUT_IN_PLACEHOLDER = 2000f;

    public static void main(String[] args) {
        float[] xValues = {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f};
        float[] yValues = new float[10];

        Random rnd = new Random(42);

        for (int i = 0; i < yValues.length; i++) {
            yValues[i] = (float) (10 * xValues[i] + 2 + 0.1 * (rnd.nextDouble() - 0.5));
        }

        try (Graph graph = new Graph()) {
            Ops tf = Ops.create(graph);

            // Define placeholders
            Placeholder<Float> X = tf.placeholder(Float.class, Placeholder.shape(Shape.scalar()));
            Placeholder<Float> Y = tf.placeholder(Float.class, Placeholder.shape(Shape.scalar()));

            // Define variables
            Variable<Float> weight = tf.variable(Shape.scalar(), Float.class);
            Variable<Float> bias = tf.variable(Shape.scalar(), Float.class);

            // Init variables
            Assign<Float> weightInit = tf.assign(weight, tf.constant(1f));
            Assign<Float> biasInit = tf.assign(bias, tf.constant(1f));

            // Define the model function weight*x + bias
            Mul<Float> mul = tf.math.mul(X, weight);
            Add<Float> yPredicted = tf.math.add(mul, bias);

            // Define MSE
            Pow<Float> sum = tf.math.pow(tf.math.sub(yPredicted, Y), tf.constant(2f));
            Div<Float> mse = tf.math.div(sum, tf.constant(2f * n));

            Gradients gradients = tf.gradients(mse, List.of(weight, bias));

            Constant<Float> alpha = tf.constant(0.2f);

            ApplyGradientDescent<Float> weightGradientDescent = tf.train.applyGradientDescent(weight, alpha, gradients.dy(0));
            ApplyGradientDescent<Float> biasGradientDescent = tf.train.applyGradientDescent(bias, alpha, gradients.dy(1));

            try (Session session = new Session(graph)) {

                // Initialize graph variables
                session.runner()
                        .addTarget(weightInit)
                        .addTarget(biasInit)
                        .run();

                // Train the model on data
                for (int i = 0; i < xValues.length; i++) {
                    float y = yValues[i];
                    float x = xValues[i];

                    Tensor<?> xTensor = Tensor.create(x);
                    Tensor<?> yTensor = Tensor.create(y);

                    session.runner()
                            .addTarget(weightGradientDescent)
                            .addTarget(biasGradientDescent)
                            .feed(X.asOutput(), xTensor)
                            .feed(Y.asOutput(), yTensor)
                            .run();

                    System.out.println("Training phase");
                    System.out.println("X is " + x + " Y is " + y);

                }


                // Extract the weight value
                float weightValue = session.runner()
                        .fetch("Variable")
                        .run().get(0).floatValue();

                System.out.println("Weight is " + weightValue);

                // Extract the bias value
                float biasValue = session.runner()
                        .fetch("Variable_1")
                        .run().get(0).floatValue();

                System.out.println("Bias is " + biasValue);


                // Let's predict y for x = 10f
                float x = 10f;
                float predictedY = 0f;

                Tensor<?> xTensor = Tensor.create(x);
                Tensor<?> yTensor = Tensor.create(NO_MEANING_VALUE_TO_PUT_IN_PLACEHOLDER);

                predictedY = session.runner()
                        .feed(X.asOutput(), xTensor)
                        .feed(Y.asOutput(), yTensor)
                        .fetch(yPredicted)
                        .run().get(0).floatValue();

                System.out.println("Predicted value: " + predictedY);
            }
        }
    }
}

