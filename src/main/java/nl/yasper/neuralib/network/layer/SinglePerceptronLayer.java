package nl.yasper.neuralib.network.layer;

import nl.yasper.neuralib.network.perceptron.Perceptron;

public class SinglePerceptronLayer<T extends Perceptron> extends PerceptronLayer<T> {

    public SinglePerceptronLayer(T perceptron) {
        super(1);
        addPerceptron(perceptron);
    }

}
