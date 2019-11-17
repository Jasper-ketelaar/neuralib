package nl.yasper.neuralib;

import nl.yasper.neuralib.display.NetworkFrame;
import nl.yasper.neuralib.network.NeuralNetwork;
import nl.yasper.neuralib.network.activation.ActivationFunction;
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

        PerceptronLayer inputLayer = new InputLayer(2);

        PerceptronLayer hidden = new LayerBuilder(2)
                .withActivationFunction(ActivationFunction.SIGMOID)
                .withLearningRate(.2)
                .withPerceptrons(2, 2)
                .build();

        PerceptronLayer output = new SinglePerceptronLayer(new LearningPerceptron(2, .2, ActivationFunction.SIGMOID));
        NeuralNetwork neuralNetwork = new NeuralNetwork(inputLayer, new PerceptronLayer[]{hidden}, output);


        Future<NetworkFrame> fn = NetworkFrame.display(neuralNetwork);
        NetworkFrame waited = fn.get(1500, TimeUnit.MILLISECONDS);

        double[][] inputs = {{0, 1}, {0, 0}, {1, 0}, {1, 1}};
        double[][] outputs = {{0}, {1}, {0}, {1}};

        //neuralNetwork.trainUntil(new double[][]{{0, 1}, {0, 0}, {1, 0}, {1, 1}}, new double[][]{{0}, {1}, {0}, {1}}, .2);
        waited.getPanel().setMousePressCallback(() -> {
            Random random = new Random();
            int train = random.nextInt(4);
            neuralNetwork.train(inputs[train], outputs[train]);
            System.out.printf("Trained %s input, expected %s output, predicted %s output \n",
                    Arrays.toString(inputs[train]), Arrays.toString(outputs[train]), Arrays.toString(neuralNetwork.predict(inputs[train])));
        });
    }
}
