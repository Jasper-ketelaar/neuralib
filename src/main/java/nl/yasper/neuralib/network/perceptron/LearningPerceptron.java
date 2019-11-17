package nl.yasper.neuralib.network.perceptron;

import nl.yasper.neuralib.activation.ActivationFunction;
import nl.yasper.neuralib.math.ArrayMath;

import java.util.Random;

public class LearningPerceptron extends Perceptron {

    private final double[] weights;
    private final double learning;

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
        weights[index] += delta;
    }

    @Override
    public double getWeightedProduct(double[] inputs) {
        return ArrayMath.dotProduct(inputs, weights);
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
