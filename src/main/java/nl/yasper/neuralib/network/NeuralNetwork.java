package nl.yasper.neuralib.network;

import nl.yasper.neuralib.network.layer.PerceptronLayer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

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

    public double train(double[] inputs, double[] outputs) {
        double[][] result = computeEpochMatrix(inputs);

        double[] prediction = result[hidden.size() + 1];
        double sqSum = 0.0;
        for (int i = 0; i < outputs.length; i++) {
            sqSum += Math.pow(outputs[i] - prediction[i], 2);
        }

        return sqSum / outputs.length;
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
