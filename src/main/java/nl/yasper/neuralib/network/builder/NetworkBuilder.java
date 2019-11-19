package nl.yasper.neuralib.network.builder;

import nl.yasper.neuralib.network.NeuralNetwork;
import nl.yasper.neuralib.network.activation.ActivationFunction;
import nl.yasper.neuralib.network.layer.InputLayer;
import nl.yasper.neuralib.network.layer.PerceptronLayer;
import nl.yasper.neuralib.network.perceptron.InputPerceptron;
import nl.yasper.neuralib.network.perceptron.LearningPerceptron;

import java.util.ArrayList;
import java.util.List;

public class NetworkBuilder {

    private PerceptronLayer<InputPerceptron> input = null;
    private PerceptronLayer<LearningPerceptron> output = null;
    private List<PerceptronLayer<LearningPerceptron>> hidden = new ArrayList<>();
    private double learningRate = 0.1;

    public NetworkBuilder withInputLayer(int size) {
        this.input = new InputLayer(size);
        return this;
    }

    public NetworkBuilder withLearningRate(double learningRate) {
        this.learningRate = learningRate;
        return this;
    }

    private int getPreviousLayerSize() {
        int prevSize;
        if (hidden.size() == 0) {
            prevSize = input.getSize();
        } else {
            prevSize = hidden.get(hidden.size() - 1).getSize();
        }

        return prevSize;
    }

    public NetworkBuilder addHiddenLayer(int size, ActivationFunction av) {
        PerceptronLayer<LearningPerceptron> hiddenLayer = new LayerBuilder(size)
                .withActivationFunction(av)
                .withLearningRate(learningRate)
                .withPerceptrons(getPreviousLayerSize(), size)
                .build();
        hidden.add(hiddenLayer);

        return this;
    }

    public NetworkBuilder withOutputLayer(int size, ActivationFunction av) {
        this.output = new LayerBuilder(size)
                .withActivationFunction(av)
                .withLearningRate(learningRate)
                .withPerceptrons(getPreviousLayerSize(), size)
                .build();

        return this;
    }

    public NeuralNetwork build() {
        return new NeuralNetwork(input, hidden, output);
    }

}
