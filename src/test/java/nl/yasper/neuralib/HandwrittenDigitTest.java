package nl.yasper.neuralib;

import nl.yasper.neuralib.network.NeuralNetwork;
import nl.yasper.neuralib.network.activation.ActivationFunction;
import nl.yasper.neuralib.network.builder.NetworkBuilder;
import org.testng.annotations.Test;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeoutException;

public class HandwrittenDigitTest {

    private static double[][] parseLine(String line) {
        String[] split = line.split(",");
        double[] result = new double[split.length];
        for (int i = 1; i < split.length; i++) {
            result[i] = Double.parseDouble(split[i]) / 255.0;
        }

        double[] output = new double[10];
        output[Integer.parseInt(split[0])] = 1;

        return new double[][]{output, result};
    }

    @Test
    public void testHandWritten() throws IOException, InterruptedException, ExecutionException, TimeoutException {
        BufferedReader csvReader = new BufferedReader(new InputStreamReader(HandwrittenDigitTest.class.getResourceAsStream("handwritten-training.csv")));
        csvReader.readLine();

        double[][] inputs = new double[42000][];
        double[][] outputs = new double[42000][];

        int counter = 0;
        String line;
        while ((line = csvReader.readLine()) != null) {
            double[][] mapping = parseLine(line);
            outputs[counter] = mapping[0];
            inputs[counter] = mapping[1];
            counter++;
        }

        NeuralNetwork neuralNetwork = new NetworkBuilder()
                .withLearningRate(.2)
                .withInputLayer(785)
                .addHiddenLayer(11, ActivationFunction.SIGMOID)
                .withOutputLayer(10, ActivationFunction.SIGMOID)
                .build();

        neuralNetwork.trainUntil(inputs, outputs, .5, 1, 1);
    }
}
