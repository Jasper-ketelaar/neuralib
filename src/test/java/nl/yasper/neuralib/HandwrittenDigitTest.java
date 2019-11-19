package nl.yasper.neuralib;

import nl.yasper.neuralib.math.ArrayMath;
import nl.yasper.neuralib.network.NeuralNetwork;
import nl.yasper.neuralib.network.activation.ActivationFunction;
import nl.yasper.neuralib.network.builder.NetworkBuilder;
import nl.yasper.neuralib.network.data.InputOutputSet;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.io.InputStream;

public class HandwrittenDigitTest {

    @Test
    public void testHandWritten() throws IOException {
        InputStream trainingStream = HandwrittenDigitTest.class.getResourceAsStream("handwritten-training.csv");
        InputOutputSet data = InputOutputSet.csvToClassifier(trainingStream, dbl -> dbl / 255.0, 1, 10);
        InputOutputSet[] split = data.split(35000);

        NeuralNetwork neuralNetwork = new NetworkBuilder()
                .withLearningRate(.04)
                .withInputLayer(784)
                .addHiddenLayer(40, ActivationFunction.SIGMOID)
                .withOutputLayer(10, ActivationFunction.SOFTMAX)
                .build();

        neuralNetwork.trainUntil(split[0].getInputs(), split[0].getOutputs(), 32, Runtime.getRuntime().availableProcessors(), (error, epoch) -> {
            return error <= 250 || epoch > 25;
        }, 1);

        int wrong = 0;
        for (int i = 0; i < split[1].getInputs().length; i++) {
            int prediction = neuralNetwork.indexPredict(split[1].getInputs()[i]);
            if (prediction != ArrayMath.getMaxIndex(split[1].getOutputs()[i])) {
                wrong++;
            }
        }

        double accuracy = (1.0 - (double) wrong / (double) split[1].getInputs().length);
        System.out.printf("On %d entries we had %d wrong which is a %.2f accuracy rate", split[1].getInputs().length, wrong,
                (1.0 - (double) wrong / (double) split[1].getInputs().length));

        Assert.assertTrue(accuracy > .95);
    }
}
