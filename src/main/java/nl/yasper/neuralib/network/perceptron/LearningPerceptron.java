package nl.yasper.neuralib.network.perceptron;

import nl.yasper.neuralib.math.ArrayMath;
import nl.yasper.neuralib.network.activation.ActivationFunction;

import java.util.Random;

public class LearningPerceptron extends Perceptron {

    private final double[] weights;
    private final double learning;
    private double bias;

    public LearningPerceptron(int inputLength, double learning, ActivationFunction activation) {
        super(inputLength, activation);
        this.weights = new double[inputLength];
        this.learning = learning;
        initializeWeights();
    }

    public double getBias() {
        return bias;
    }

    private void initializeWeights() {
        Random random = new Random();
        double range = 2.4 / (double) weights.length;
        for (int i = 0; i < weights.length; i++) {
            this.weights[i] = 2 * random.nextDouble() * range - range;
        }

        this.bias = 2 * random.nextDouble() * range - range;;
    }

    public void updateBias(double delta) {
        bias -= learning * delta;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public void train(double[] inputs, double output) {
        double prediction = predict(inputs);
        double error = output - prediction;

        for (int i = 0; i < getInputLength(); i++) {
            double delta = learning * inputs[i] * error;
            updateWeight(i, delta);
        }
    }

    public void updateWeight(int index, double delta) {
        weights[index] += learning * delta;
    }

    public void setWeight(int index, double weight) {
        weights[index] = weight;
    }

    @Override
    public double getWeightedProduct(double[] inputs) {
        return ArrayMath.dotProduct(inputs, weights) - bias;
    }

    @Override
    public LearningPerceptron createNew() {
        return new LearningPerceptron(getInputLength(), learning, getActivation());
    }

    public double[] getWeights() {
        return weights;
    }

    public double getLearningRate() {
        return learning;
    }
}
