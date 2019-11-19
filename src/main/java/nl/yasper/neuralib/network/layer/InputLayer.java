package nl.yasper.neuralib.network.layer;

import nl.yasper.neuralib.network.perceptron.InputPerceptron;
import nl.yasper.neuralib.network.perceptron.Perceptron;

import java.security.InvalidParameterException;

public class InputLayer extends PerceptronLayer<InputPerceptron> {

    public InputLayer(int size) {
        super(size);
        addPerceptron(new InputPerceptron(), size);
    }

    @Override
    public double[] predict(double[] input) {
        double[] result = new double[getSize()];
        for (int i = 0; i < getSize(); i++) {
            result[i] = getPerceptron(i).predict(input[i]);
        }

        return result;
    }

    @Override
    public PerceptronLayer<InputPerceptron> clone() {
        return new InputLayer(getSize());
    }
}
