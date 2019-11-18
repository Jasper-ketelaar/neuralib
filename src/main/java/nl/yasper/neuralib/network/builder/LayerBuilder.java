package nl.yasper.neuralib.network.builder;

import nl.yasper.neuralib.network.activation.ActivationFunction;
import nl.yasper.neuralib.network.perceptron.LearningPerceptron;
import nl.yasper.neuralib.network.layer.PerceptronLayer;
import nl.yasper.neuralib.network.perceptron.Perceptron;

public class LayerBuilder {

    private final PerceptronLayer<LearningPerceptron> layer;

    private ActivationFunction activation = ActivationFunction.IDENTITY;
    private double learning = 0.2;

    public LayerBuilder(int size) {
        this.layer = new PerceptronLayer<>(size);
    }

    public LayerBuilder withActivationFunction(ActivationFunction activation) {
        this.activation = activation;
        return this;
    }

    public LayerBuilder withLearningRate(double learningRate) {
        this.learning = learningRate;
        return this;
    }

    public LayerBuilder withPerceptrons(int inputLength, int amount, ActivationFunction customActivation) {
        layer.addPerceptron(new LearningPerceptron(inputLength, learning, customActivation), amount);
        return this;
    }

    public LayerBuilder withPerceptrons(int inputLength, int amount) {
        return withPerceptrons(inputLength, amount, activation);
    }

    public PerceptronLayer<LearningPerceptron> build() {
        return layer;
    }

}
