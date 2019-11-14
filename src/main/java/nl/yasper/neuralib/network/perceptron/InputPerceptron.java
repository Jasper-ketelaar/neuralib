package nl.yasper.neuralib.network.perceptron;

import nl.yasper.neuralib.activation.ActivationFunction;

public class InputPerceptron extends Perceptron {

    public InputPerceptron() {
        super(1, ActivationFunction.IDENTITY);
    }

    @Override
    public double getWeightedProduct(double[] inputs) {
        return inputs[0];
    }

    @Override
    public Perceptron clone() {
        return new InputPerceptron();
    }
}
