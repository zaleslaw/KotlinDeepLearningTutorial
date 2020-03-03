import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.function.Function;

public class TFLoader {
    public static void main(String[] args) throws IOException {
        System.out.println(TensorFlow.version());

        List<MnistUtils.MnistLabeledImage> images = MnistUtils.mnistAsList(
                "src/main/resources/datasets/t10k-images-idx3-ubyte",
                "src/main/resources/datasets/t10k-labels-idx1-ubyte",
                new Random(0),
                10000
        );

        Function<double[], Tensor<?>> reshaper = doubles -> {
            float[][][] reshaped = new float[1][28][28];
            for (int i = 0; i < doubles.length; i++)
                reshaped[0][i / 28][i % 28] = (float) doubles[i];
            return Tensor.create(reshaped);
        };

        // load the model Bundle
        try (SavedModelBundle b = SavedModelBundle.load("src/main/resources/models", "serve")) {
            // create the session from the Bundle
            Session sess = b.session();
            Session.Runner runner = sess.runner();

            for (MnistUtils.MnistLabeledImage image : images) {
                long[] res = runner.feed("Placeholder", reshaper.apply(image.getPixels()))
                        .fetch("ArgMax")
                        .run()
                        .get(0)
                        .copyTo(new long[1]);
                System.out.println(res[0]);
            }
        }
    }
}


