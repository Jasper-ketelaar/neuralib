package nl.yasper.neuralib;

import nl.yasper.neuralib.display.NetworkFrame;
import nl.yasper.neuralib.network.NeuralNetwork;
import nl.yasper.neuralib.network.activation.ActivationFunction;
import nl.yasper.neuralib.network.builder.LayerBuilder;
import nl.yasper.neuralib.network.layer.InputLayer;
import nl.yasper.neuralib.network.layer.PerceptronLayer;
import nl.yasper.neuralib.network.perceptron.LearningPerceptron;

import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;

public class NeuraLibTest {

    public static void main(String[] args) throws InterruptedException, ExecutionException, TimeoutException {
        PerceptronLayer inputLayer = new InputLayer(2);

        PerceptronLayer<LearningPerceptron> hidden = new LayerBuilder(2)
                .withActivationFunction(ActivationFunction.SIGMOID)
                .withLearningRate(.1)
                .withPerceptrons(2, 2)
                .build();

        PerceptronLayer<LearningPerceptron> output = new LayerBuilder(1)
                .withActivationFunction(ActivationFunction.SIGMOID)
                .withLearningRate(.1)
                .withPerceptrons(2, 1)
                .build();

        NeuralNetwork neuralNetwork = new NeuralNetwork(inputLayer, new PerceptronLayer[]{hidden}, output);

        double[][] inputs = {{1, 1}, {0, 1}, {1, 0}, {0, 0}};
        double[][] outputs = {{0}, {1}, {1}, {0}};

        Future<NetworkFrame> fn = NetworkFrame.display(neuralNetwork);
        NetworkFrame waited = fn.get(1500, TimeUnit.MILLISECONDS);

        AtomicBoolean cancel = new AtomicBoolean(false);


        new Thread(() -> {
            Random random = new Random();
            double sse = 1;
            double max_sse = Double.MIN_VALUE;
            int epoch = 1;
            while (sse > 0.001 && !cancel.get()) {
                sse = 0.0;
                int start = random.nextInt(4);
                for (int i = 0; i < inputs.length; i++) {
                    double err = neuralNetwork.train(inputs[(start + i) % 4], outputs[(start + i) % 4]);
                    sse += err;
                }

                if ((epoch % 100) == 0) {
                    System.out.printf("Epoch %d: SSE=%.6f\n", epoch, sse);
                }

                epoch += 1;
            }
        }).start();

        waited.getPanel().setMousePressCallback(() -> {
            cancel.set(true);
            System.out.println();
            System.out.println(Arrays.toString(neuralNetwork.predict(inputs[0])));
            System.out.println(Arrays.toString(neuralNetwork.predict(inputs[1])));
            System.out.println(Arrays.toString(neuralNetwork.predict(inputs[2])));
            System.out.println(Arrays.toString(neuralNetwork.predict(inputs[3])));
        });
    }

    public static void braille(String[] args) throws InterruptedException, ExecutionException, TimeoutException {

        PerceptronLayer inputLayer = new InputLayer(6);

        PerceptronLayer hidden = new LayerBuilder(5)
                .withActivationFunction(ActivationFunction.SIGMOID)
                .withLearningRate(.2)
                .withPerceptrons(6, 5)
                .build();

        PerceptronLayer output = new LayerBuilder(30)
                .withActivationFunction(ActivationFunction.SIGMOID)
                .withLearningRate(.2)
                .withPerceptrons(5, 30)
                .build();

        NeuralNetwork neuralNetwork = new NeuralNetwork(inputLayer, new PerceptronLayer[]{hidden}, output);


        Future<NetworkFrame> fn = NetworkFrame.display(neuralNetwork);
        NetworkFrame waited = fn.get(1500, TimeUnit.MILLISECONDS);

        double[][] inputs = {
                {1, 0, 0, 0, 0, 0},
                {1, 0, 1, 0, 0, 0},
                {1, 0, 1, 0, 1, 1},
                {1, 1, 1, 1, 0, 0},
                {1, 1, 0, 1, 0, 0},
                {1, 1, 0, 1, 0, 1},
                {1, 0, 0, 1, 0, 0},
                {0, 1, 1, 0, 1, 1},
                {1, 0, 0, 1, 1, 1},
                {0, 1, 1, 0, 0, 0},
                {0, 1, 1, 1, 0, 0},
                {1, 0, 0, 0, 1, 0},
                {1, 0, 1, 0, 1, 0},
                {1, 0, 1, 0, 0, 1},
                {1, 1, 0, 0, 1, 0},
                {1, 1, 0, 1, 1, 0},
                {1, 1, 1, 0, 0, 1},
                {1, 0, 0, 1, 1, 0},
                {1, 1, 1, 0, 1, 0},
                {1, 0, 1, 1, 1, 0},
                {0, 1, 1, 0, 1, 0},
                {0, 1, 1, 1, 1, 0},
                {1, 1, 0, 0, 0, 1},
                {1, 0, 0, 0, 1, 1},
                {1, 1, 1, 0, 0, 0},
                {1, 0, 1, 1, 0, 0},
                {1, 1, 0, 0, 0, 0},
                {1, 0, 0, 0, 0, 1},
                {1, 1, 1, 1, 0, 1},
                {1, 0, 0, 1, 0, 1},
        };


        double[][] outputs = new double[30][30];
        for (int i = 0; i < outputs.length; i++) {
            outputs[i] = out(i);
        }

        double mse = 1;
        while (mse >= 0.03) {
            Random random = new Random();
            int train = random.nextInt(inputs.length);
            System.out.println((mse = neuralNetwork.train(inputs[train], outputs[train])));
        }

        waited.getPanel().setMousePressCallback(() -> {
            int train = new Random().nextInt(inputs.length);
            int highest = 0;
            double[] res = neuralNetwork.predict(inputs[train]);
            for (int i = 0; i < res.length; i++) {
                res[i] = Math.round(res[i] * 100.0) / 100.0;
            }

            System.out.printf("input %s \n expected output: %d \n result: %s \n\n",
                    Arrays.toString(inputs[train]), train, Arrays.toString(res));

        });
    }

    private static double[] out(int index) {
        double[] result = new double[30];
        result[index] = 1;
        return result;
    }
}
