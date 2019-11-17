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

    private void initializeWeights() {
        Random random = new Random();
        for (int i = 0; i < weights.length; i++) {
            this.weights[i] = random.nextDouble();
        }

        this.bias = random.nextDouble();
    }

    public void updateBias(double delta) {
        bias += learning * delta;
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
        //System.out.printf("Weight %d changed by %.2f\n", index, delta * learning);
        weights[index] += learning * delta;
    }

    @Override
    public double getWeightedProduct(double[] inputs) {
        return ArrayMath.dotProduct(inputs, weights) ;
    }

    @Override
    public Perceptron clone() {
        return new LearningPerceptron(getInputLength(), learning, getActivation());
    }

    public double[] getWeights() {
        return weights;
    }

    public double getLearningRate() {
        return learning;
    }
}
