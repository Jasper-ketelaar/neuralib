package nl.yasper.neuralib;

import nl.yasper.neuralib.network.NeuralNetwork;
import nl.yasper.neuralib.network.activation.ActivationFunction;
import nl.yasper.neuralib.network.builder.LayerBuilder;
import nl.yasper.neuralib.network.layer.InputLayer;
import nl.yasper.neuralib.network.layer.PerceptronLayer;
import org.testng.Assert;
import org.testng.annotations.Test;

public class BrailleTest {

    @Test
    public void testBrailleAlphabet() {
        PerceptronLayer inputLayer = new InputLayer(6);

        PerceptronLayer hidden = new LayerBuilder(5)
                .withActivationFunction(ActivationFunction.SIGMOID)
                .withLearningRate(.2)
                .withPerceptrons(6, 5)
                .build();

        PerceptronLayer output = new LayerBuilder(30)
                .withActivationFunction(ActivationFunction.SIGMOID)
                .withLearningRate(.2)
                .withPerceptrons(5, 30)
                .build();

        NeuralNetwork neuralNetwork = new NeuralNetwork(inputLayer, new PerceptronLayer[]{hidden}, output);


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

        neuralNetwork.trainUntil(inputs, outputs, .1);
        for (int i = 0; i < outputs.length; i++) {
            Assert.assertEquals(neuralNetwork.binaryPredict(inputs[i]), outputs[i]);
        }
    }

    private double[] out(int index) {
        double[] result = new double[30];
        result[index] = 1;
        return result;
    }
}
