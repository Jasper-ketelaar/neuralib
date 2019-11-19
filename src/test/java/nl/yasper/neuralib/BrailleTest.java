package nl.yasper.neuralib;

import nl.yasper.neuralib.network.NeuralNetwork;
import nl.yasper.neuralib.network.activation.ActivationFunction;
import nl.yasper.neuralib.network.builder.LayerBuilder;
import nl.yasper.neuralib.network.builder.NetworkBuilder;
import nl.yasper.neuralib.network.layer.InputLayer;
import nl.yasper.neuralib.network.layer.PerceptronLayer;
import org.testng.Assert;
import org.testng.annotations.Test;

public class BrailleTest {

    @Test
    public void testBrailleAlphabet() {
        NeuralNetwork neuralNetwork = new NetworkBuilder()
                .withLearningRate(.2)
                .withInputLayer(6)
                .addHiddenLayer(5, ActivationFunction.SIGMOID)
                .withOutputLayer(30, ActivationFunction.SIGMOID)
                .build();

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
