package nl.yasper.neuralib.network.layer;

import nl.yasper.neuralib.network.perceptron.Perceptron;

public class SinglePerceptronLayer extends PerceptronLayer {

    public SinglePerceptronLayer(Perceptron perceptron) {
        super(1);
        addPerceptron(perceptron);
    }

}
