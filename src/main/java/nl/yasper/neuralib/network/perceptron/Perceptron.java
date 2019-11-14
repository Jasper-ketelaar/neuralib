package nl.yasper.neuralib.network.perceptron;

import nl.yasper.neuralib.activation.ActivationFunction;

import java.security.InvalidParameterException;

public abstract class Perceptron {

    private final int inputLength;
    private final ActivationFunction activation;

    public Perceptron(int inputLength, ActivationFunction activation) {
        this.inputLength = inputLength;
        this.activation = activation;
    }

    public int getInputLength() {
        return inputLength;
    }

    public double predict(double[] inputs) {
        if (inputs.length != getInputLength()) {
            throw new InvalidParameterException(String.format("Input length was supposed to be %d but we got %d",
                    getInputLength(), inputs.length));
        }

        double product = getWeightedProduct(inputs);
        return getActivation().activate(product);
    }

    public double predict(double input) {
        return predict(new double[]{input});
    }

    public abstract double getWeightedProduct(double[] inputs);

    public abstract Perceptron clone();

    public ActivationFunction getActivation() {
        return activation;
    }

    public double getActivationValue(double current) {
        return getActivation().activate(current);
    }

}