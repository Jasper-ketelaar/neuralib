package nl.yasper.neuralib.network.layer;

import nl.yasper.neuralib.network.perceptron.InputPerceptron;
import nl.yasper.neuralib.network.perceptron.Perceptron;

import java.security.InvalidParameterException;

public class InputLayer extends PerceptronLayer {

    public InputLayer(int size) {
        super(size);
        addPerceptron(new InputPerceptron(), size);
    }

    @Override
    public void addPerceptron(Perceptron perceptron, int amount) {
        if (perceptron instanceof InputPerceptron) {
            super.addPerceptron(perceptron, amount);
        } else {
            throw new InvalidParameterException("An input layer can only contain input perceptrons");
        }
    }

    @Override
    public double[] predict(double[] input) {
        double[] result = new double[getSize()];
        for (int i = 0; i < getSize(); i++) {
            result[i] = getPerceptron(i).predict(input[i]);
        }

        return result;
    }

}
