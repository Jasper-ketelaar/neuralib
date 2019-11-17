package nl.yasper.neuralib;

import nl.yasper.neuralib.activation.ActivationFunction;
import nl.yasper.neuralib.activation.Threshold;
import nl.yasper.neuralib.network.NeuralNetwork;
import nl.yasper.neuralib.network.builder.LayerBuilder;
import nl.yasper.neuralib.network.layer.InputLayer;
import nl.yasper.neuralib.network.layer.PerceptronLayer;
import nl.yasper.neuralib.network.layer.SinglePerceptronLayer;
import nl.yasper.neuralib.network.perceptron.LearningPerceptron;

import java.util.Arrays;
import java.util.Random;

public class NeuraLibTest {

    public static void main(String[] args) {

        PerceptronLayer inputLayer = new InputLayer(2);

        PerceptronLayer hidden = new LayerBuilder(1)
                .withActivationFunction(ActivationFunction.SIGMOID)
                .withLearningRate(.1)
                .withPerceptrons(2, 1)
                .build();

        PerceptronLayer output = new SinglePerceptronLayer(new LearningPerceptron(2, .1, new Threshold(.4)));
        NeuralNetwork neuralNetwork = new NeuralNetwork(inputLayer, new PerceptronLayer[]{}, output);

        for (int i = 0; i < 100; i++) {
            Random r = new Random();
            int x = r.nextInt(2);
            int y = r.nextInt(2);
            int z = (x & y) == 1 ? 1 : 0;
            neuralNetwork.train(new double[]{x, y}, new double[]{z});
        }

        System.out.println(Arrays.toString(neuralNetwork.predict(new double[]{0, 0})));
    }
}
