package nl.yasper.neuralib.network;

import nl.yasper.neuralib.network.layer.InputLayer;
import nl.yasper.neuralib.network.layer.PerceptronLayer;
import nl.yasper.neuralib.network.perceptron.LearningPerceptron;
import nl.yasper.neuralib.network.perceptron.Perceptron;

import java.util.*;

public class NeuralNetwork {

    private final PerceptronLayer input;
    private final List<PerceptronLayer> hidden;
    private final PerceptronLayer output;

    public NeuralNetwork(int input, int[] hidden, int output) {
        this.input = new PerceptronLayer(input);
        this.hidden = new ArrayList<>(hidden.length);
        for (int value : hidden) {
            this.hidden.add(new PerceptronLayer(value));
        }
        this.output = new PerceptronLayer(output);
    }

    public NeuralNetwork(PerceptronLayer input, PerceptronLayer[] hidden, PerceptronLayer output) {
        this.input = input;
        this.hidden = Arrays.asList(hidden);
        this.output = output;
    }

    public NeuralNetwork(PerceptronLayer input, PerceptronLayer output) {
        this(input, new PerceptronLayer[0], output);
    }

    public void train(double[] inputs, double[] outputs) {
        double[][] result = computeEpochMatrix(inputs);
        double[] predictions = result[hidden.size() + 1];

        Map<PerceptronLayer, double[]> errorMap = backPropogatedError(predictions, outputs);
        PerceptronLayer current = output;
        int layerIndex = hidden.size() + 1;
        while (!(current instanceof InputLayer)) {
            double[] errors = errorMap.get(current);
            for (int index = 0; index < current.getSize(); index++) {
                LearningPerceptron lp = (LearningPerceptron) current.getPerceptron(index);
                for (int i = 0; i < lp.getInputLength(); i++) {
                    double delta = result[layerIndex][index] * lp.getLearningRate() * errors[index];
                    lp.updateWeight(i, delta);
                }
            }

            current = getPrevious(current);
            layerIndex--;
        }
    }

    private Map<PerceptronLayer, double[]> backPropogatedError(double[] predictions, double[] expected) {
        Map<PerceptronLayer, double[]> errorMap = new HashMap<>();
        double[] outputErrors = new double[predictions.length];
        for (int i = 0; i < predictions.length; i++) {
            outputErrors[i] = expected[i] - predictions[i];
        }

        errorMap.put(output, outputErrors);

        PerceptronLayer previousLayer = output;
        for (int i = hidden.size() - 1; i >= 0; i--) {
            PerceptronLayer currentLayer = hidden.get(i);
            double[] prevLayerErrors = errorMap.get(previousLayer);
            double[] layerErrors = new double[currentLayer.getSize()];
            for (int index = 0; index < currentLayer.getSize(); index++) {
                for (int x = 0; x < prevLayerErrors.length; x++) {
                    LearningPerceptron currPrevPerceptron = (LearningPerceptron) previousLayer.getPerceptron(x);
                    layerErrors[index] += prevLayerErrors[x] * currPrevPerceptron.getWeights()[index];
                }
            }

            errorMap.put(currentLayer, layerErrors);
            previousLayer = currentLayer;
        }

        return errorMap;
    }


    public PerceptronLayer getPrevious(PerceptronLayer from) {
        if (from == output) {
            if (hidden.size() == 0) {
                return input;
            } else {
                return hidden.get(hidden.size() - 1);
            }
        } else if (from == input) {
            return null;
        }

        PerceptronLayer previous = hidden.get(0);
        if (previous == from) {
            return input;
        }

        for (PerceptronLayer hiddenLayer : hidden) {
            if (hiddenLayer == from) {
                return previous;
            }

            previous = hiddenLayer;
        }

        return null;
    }

    public double[] predict(double[] inputs) {
        return computeEpochMatrix(inputs)[hidden.size() + 1];
    }

    private double[][] computeEpochMatrix(double[] inputs) {
        double[][] result = new double[2 + hidden.size()][];
        result[0] = input.predict(inputs);
        for (int i = 0; i < hidden.size(); i++) {
            result[i + 1] = hidden.get(i).predict(result[i]);
        }


        result[hidden.size() + 1] = output.predict(result[hidden.size()]);
        return result;
    }

    public PerceptronLayer getInputLayer() {
        return input;
    }

    public PerceptronLayer getOutputLayer() {
        return output;
    }

    public List<PerceptronLayer> getHiddenLayers() {
        return hidden;
    }

}
