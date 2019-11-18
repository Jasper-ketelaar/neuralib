package nl.yasper.neuralib;

import nl.yasper.neuralib.network.NeuralNetwork;
import nl.yasper.neuralib.network.activation.ActivationFunction;
import nl.yasper.neuralib.network.builder.LayerBuilder;
import nl.yasper.neuralib.network.layer.InputLayer;
import nl.yasper.neuralib.network.layer.PerceptronLayer;
import nl.yasper.neuralib.network.perceptron.LearningPerceptron;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.Random;

public class XORTest {

    @Test
    public void testXORNetwork() {
        System.out.println("Testing XOR network");
        PerceptronLayer inputLayer = new InputLayer(2);

        PerceptronLayer<LearningPerceptron> hidden = new LayerBuilder(2)
                .withActivationFunction(ActivationFunction.SIGMOID)
                .withLearningRate(.1)
                .withPerceptrons(2, 2)
                .build();

        PerceptronLayer<LearningPerceptron> output = new LayerBuilder(1)
                .withActivationFunction(ActivationFunction.SIGMOID)
                .withLearningRate(.1)
                .withPerceptrons(2, 1)
                .build();

        NeuralNetwork neuralNetwork = new NeuralNetwork(inputLayer, new PerceptronLayer[]{hidden}, output);

        double[][] inputs = {{1, 1}, {0, 1}, {1, 0}, {0, 0}};
        double[][] outputs = {{0}, {1}, {1}, {0}};

        neuralNetwork.trainUntil(inputs, outputs, 0.001);


        Assert.assertEquals(neuralNetwork.binaryPredict(new double[]{0, 1}), new double[]{1});
        Assert.assertEquals(neuralNetwork.binaryPredict(new double[]{1, 0}), new double[]{1});
        Assert.assertEquals(neuralNetwork.binaryPredict(new double[]{0, 0}), new double[]{0});
        Assert.assertEquals(neuralNetwork.binaryPredict(new double[]{1, 1}), new double[]{0});
    }
}
