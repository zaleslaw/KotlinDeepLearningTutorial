package tensorflow;

import org.tensorflow.*;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;
import util.MnistUtils;

import java.io.IOException;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

public class Example_1_TF_Inference {
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

        try (EagerSession session = EagerSession.create()) {
            Ops tf = Ops.create(session);

            Constant<Long> a = tf.constant(10L);
            System.out.println(a.asOutput().tensor().longValue());
        }
    }
}


