package nl.yasper.neuralib;

import nl.yasper.neuralib.display.NetworkFrame;
import nl.yasper.neuralib.network.NeuralNetwork;
import nl.yasper.neuralib.network.activation.ActivationFunction;
import nl.yasper.neuralib.network.builder.LayerBuilder;
import nl.yasper.neuralib.network.layer.InputLayer;
import nl.yasper.neuralib.network.layer.PerceptronLayer;
import nl.yasper.neuralib.network.perceptron.InputPerceptron;
import nl.yasper.neuralib.network.perceptron.LearningPerceptron;
import org.testng.annotations.Test;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

public class HandwrittenDigitTest {

    public static void main(String[] args) throws IOException, InterruptedException, ExecutionException, TimeoutException {
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

        PerceptronLayer<InputPerceptron> inputLayer = new InputLayer(785);

        PerceptronLayer<LearningPerceptron> hidden = new LayerBuilder(33)
                .withActivationFunction(ActivationFunction.SIGMOID)
                .withLearningRate(.5)
                .withPerceptrons(785, 33)
                .build();

        PerceptronLayer<LearningPerceptron> hidden2 = new LayerBuilder(11)
                .withActivationFunction(ActivationFunction.SIGMOID)
                .withLearningRate(.5)
                .withPerceptrons(33, 11)
                .build();

        PerceptronLayer<LearningPerceptron> output = new LayerBuilder(10)
                .withActivationFunction(ActivationFunction.SIGMOID)
                .withLearningRate(.5)
                .withPerceptrons(11, 10)
                .build();


        NeuralNetwork neuralNetwork = new NeuralNetwork(inputLayer, Arrays.asList(hidden, hidden2), output);
        neuralNetwork.trainUntil(inputs, outputs, .5, 1);
    }

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

        PerceptronLayer inputLayer = new InputLayer(784);

        PerceptronLayer hidden = new LayerBuilder(11)
                .withActivationFunction(ActivationFunction.SIGMOID)
                .withLearningRate(.2)
                .withPerceptrons(784, 11)
                .build();

        PerceptronLayer output = new LayerBuilder(10)
                .withActivationFunction(ActivationFunction.SIGMOID)
                .withLearningRate(.2)
                .withPerceptrons(11, 10)
                .build();


        NeuralNetwork neuralNetwork = new NeuralNetwork(inputLayer, new PerceptronLayer[]{hidden}, output);

        Future<NetworkFrame> networkPanel = NetworkFrame.display(neuralNetwork);
        networkPanel.get(1500, TimeUnit.MILLISECONDS);
        //neuralNetwork.trainUntil(inputs, outputs, .5);
    }
}
