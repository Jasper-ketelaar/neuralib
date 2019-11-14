package nl.yasper.neuralib;

import nl.yasper.neuralib.activation.Sigmoid;
import nl.yasper.neuralib.activation.Threshold;
import nl.yasper.neuralib.display.NetworkPanel;
import nl.yasper.neuralib.network.NeuralNetwork;
import nl.yasper.neuralib.network.builder.LayerBuilder;
import nl.yasper.neuralib.network.layer.InputLayer;
import nl.yasper.neuralib.network.layer.PerceptronLayer;
import nl.yasper.neuralib.network.layer.SinglePerceptronLayer;
import nl.yasper.neuralib.network.perceptron.LearningPerceptron;

import javax.swing.*;

public class NeuraLibTest {

    public static void main(String[] args) {

        PerceptronLayer inputLayer = new InputLayer(40);

        PerceptronLayer hidden = new LayerBuilder(8)
                .withActivationFunction(new Sigmoid())
                .withLearingRate(.1)
                .withPerceptrons(40, 8)
                .build();

        PerceptronLayer output = new SinglePerceptronLayer(new LearningPerceptron(hidden.getSize(), .1, new Threshold(.5)));
        NeuralNetwork neuralNetwork = new NeuralNetwork(inputLayer, new PerceptronLayer[]{hidden}, output);

        SwingUtilities.invokeLater(() -> {
            NetworkPanel panel = new NetworkPanel(neuralNetwork);
            JFrame show = new JFrame("Neural network frame");
            show.setContentPane(panel);
            show.pack();
            show.setVisible(true);
        });

    }
}
