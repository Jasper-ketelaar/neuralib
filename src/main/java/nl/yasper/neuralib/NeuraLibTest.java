package nl.yasper.neuralib;

import nl.yasper.neuralib.display.NetworkFrame;
import nl.yasper.neuralib.network.NeuralNetwork;
import nl.yasper.neuralib.network.activation.ActivationFunction;
import nl.yasper.neuralib.network.activation.Sigmoid;
import nl.yasper.neuralib.network.activation.Threshold;
import nl.yasper.neuralib.network.builder.LayerBuilder;
import nl.yasper.neuralib.network.layer.InputLayer;
import nl.yasper.neuralib.network.layer.PerceptronLayer;
import nl.yasper.neuralib.network.layer.SinglePerceptronLayer;
import nl.yasper.neuralib.network.perceptron.LearningPerceptron;

import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

public class NeuraLibTest {

    public static void main(String[] args) throws InterruptedException, ExecutionException, TimeoutException {

        PerceptronLayer inputLayer = new InputLayer(6);

        PerceptronLayer hidden = new LayerBuilder(13)
                .withActivationFunction(ActivationFunction.SIGMOID)
                .withLearningRate(.2)
                .withPerceptrons(6, 13)
                .build();

        PerceptronLayer output = new LayerBuilder(30)
                .withActivationFunction(new Sigmoid())
                .withLearningRate(.2)
                .withPerceptrons(13, 30)
                .build();

        NeuralNetwork neuralNetwork = new NeuralNetwork(inputLayer, new PerceptronLayer[]{hidden}, output);


        Future<NetworkFrame> fn = NetworkFrame.display(neuralNetwork);
        NetworkFrame waited = fn.get(1500, TimeUnit.MILLISECONDS);

        double[][] inputs = {
                {1, 0, 0, 0, 0, 0},
                {1, 0, 1, 0, 0, 0},
                {1, 0, 1, 0, 1, 1},
                {1, 1, 1, 1, 0, 0},
                {1, 1, 0, 1, 0, 0},
                {1, 1, 0, 1, 0, 1},
                {1, 0, 0, 1, 0, 0},
                {0, 1, 1, 0, 1, 1},
                {1, 0, 0, 1, 1, 1},
                {0, 1, 1, 0, 0, 0},
                {0, 1, 1, 1, 0, 0},
                {1, 0, 0, 0, 1, 0},
                {1, 0, 1, 0, 1, 0},
                {1, 0, 1, 0, 0, 1},
                {1, 1, 0, 0, 1, 0},
                {1, 1, 0, 1, 1, 0},
                {1, 1, 1, 0, 0, 1},
                {1, 0, 0, 1, 1, 0},
                {1, 1, 1, 0, 1, 0},
                {1, 0, 1, 1, 1, 0},
                {0, 1, 1, 0, 1, 0},
                {0, 1, 1, 1, 1, 0},
                {1, 1, 0, 0, 0, 1},
                {1, 0, 0, 0, 1, 1},
                {1, 1, 1, 0, 0, 0},
                {1, 0, 1, 1, 0, 0},
                {1, 1, 0, 0, 0, 0},
                {1, 0, 0, 0, 0, 1},
                {1, 1, 1, 1, 0, 1},
                {1, 0, 0, 1, 0, 1},
        };


        double[][] outputs = new double[30][30];
        for (int i = 0; i < outputs.length; i++) {
            outputs[i] = out(i);
        }

        for (int epoch = 0; epoch < 1000; epoch++) {
            Random random = new Random();
            int train = random.nextInt(inputs.length);
            System.out.println(neuralNetwork.train(inputs[train], outputs[train]));
        }

        waited.getPanel().setMousePressCallback(() -> {
            for (int i = 0; i < inputs.length; i++) {
                int train = i;
                neuralNetwork.train(inputs[train], outputs[train]);
                System.out.printf("input %s \n expected output: %s \n result: %s \n\n",
                        Arrays.toString(inputs[train]), Arrays.toString(outputs[train]), (Arrays.toString(neuralNetwork.predict(inputs[train]))));
            }
        });
    }

    private static double[] out(int index) {
        double[] result = new double[30];
        result[index] = 1;
        return result;
    }
}
