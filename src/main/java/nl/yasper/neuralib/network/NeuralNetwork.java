package nl.yasper.neuralib.network;

import nl.yasper.neuralib.math.ArrayMath;
import nl.yasper.neuralib.network.activation.ActivationFunction;
import nl.yasper.neuralib.network.layer.InputLayer;
import nl.yasper.neuralib.network.layer.PerceptronLayer;
import nl.yasper.neuralib.network.perceptron.InputPerceptron;
import nl.yasper.neuralib.network.perceptron.LearningPerceptron;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicReference;

public class NeuralNetwork {

    private final PerceptronLayer<InputPerceptron> input;
    private final List<PerceptronLayer<LearningPerceptron>> hidden;
    private final PerceptronLayer<LearningPerceptron> output;

    public NeuralNetwork(int input, int[] hidden, int output) {
        this.input = new PerceptronLayer<>(input);
        this.hidden = new ArrayList<>(hidden.length);
        for (int value : hidden) {
            this.hidden.add(new PerceptronLayer<>(value));
        }
        this.output = new PerceptronLayer<>(output);
    }

    public NeuralNetwork(PerceptronLayer<InputPerceptron> input, PerceptronLayer<LearningPerceptron>[] hidden, PerceptronLayer<LearningPerceptron> output) {
        this.input = input;
        this.hidden = Arrays.asList(hidden);
        this.output = output;
    }

    public NeuralNetwork(PerceptronLayer<InputPerceptron> input, List<PerceptronLayer<LearningPerceptron>> hidden, PerceptronLayer<LearningPerceptron> output) {
        this.input = input;
        this.hidden = hidden;
        this.output = output;
    }

    private NeuralNetwork(NeuralNetwork copy) {
        this.input = copy.input.clone();
        this.output = copy.output.clone();
        this.hidden = new ArrayList<>();
        for (PerceptronLayer p : copy.hidden) {
            hidden.add(p.clone());
        }
    }

    public NeuralNetwork(PerceptronLayer input, PerceptronLayer output) {
        this(input, new PerceptronLayer[0], output);
    }

    public void trainUntil(double[][] inputs, double[][] outputs, double error, int printEpochs) {
        double sse = Double.MAX_VALUE;
        int epoch = 0;
        while (sse > error) {
            sse = 0;

            for (int i = 0; i < inputs.length; i++) {
                double[] inputSes = inputs[i];
                double[] outputSes = outputs[i];
                sse += train(inputSes, outputSes);
            }

            if ((epoch % printEpochs) == 0) {
                System.out.printf("Epoch %d: SSE=%.6f\n", epoch, sse);
            }

            epoch++;
        }
    }

    public void trainUntil(double[][] inputs, double[][] outputs, double error) {
        trainUntil(inputs, outputs, error, 10000);
    }

    public void trainUntilParallel(double[][] inputs, double[][] outputs, double error, int printEpochs, int threadPool) {
        double sse = Double.MAX_VALUE;
        int epoch = 0;
        while (sse > error) {
            sse = trainParallel(inputs, outputs, threadPool);
            if ((epoch % printEpochs) == 0) {
                System.out.printf("Epoch %d: SSE=%.6f\n", epoch, sse);
            }

            epoch++;
        }
    }

    public double batchTrain(double[][] inputs, double[][] outputs) {
        Map<PerceptronLayer, double[]> errorMap = new HashMap<>();
        for (int i = 0; i < inputs.length; i++) {
            double[][] result = computeEpochMatrix(inputs[i]);
            double[] predictions = result[hidden.size() + 1];
            Map<PerceptronLayer, double[]> currErrMap = backPropogatedErrorGradient(predictions, outputs[i], result);
            for (PerceptronLayer entry : currErrMap.keySet()) {
                if (!errorMap.containsKey(entry)) {
                    errorMap.put(entry, currErrMap.get(entry));
                } else {
                    double[] accumulatedError = errorMap.get(entry);
                    errorMap.put(entry, ArrayMath.add(accumulatedError, currErrMap.get(entry)));
                }
            }
        }

        return 0;
    }

    public double trainParallel(double[][] inputs, double[][] outputs, int threadPool) {
        Executor executor = Executors.newFixedThreadPool(threadPool);
        CompletionService<Double> completionService =
                new ExecutorCompletionService<>(executor);

        NeuralNetwork[] networks = new NeuralNetwork[threadPool];
        for (int i = 0; i < networks.length; i++) {
            networks[i] = new NeuralNetwork(this);
        }

        AtomicReference<Double> sse = new AtomicReference<>();
        for (int index = 0; index < inputs.length; index += threadPool) {
            for (int i = 0; i < threadPool; i++) {
                int batch = i;
                int dataIndex = index + batch;
                if (dataIndex >= inputs.length) {
                    break;
                }


                completionService.submit(() -> networks[batch].train(inputs[dataIndex], outputs[dataIndex]));
            }

            int received = 0;
            boolean errors = false;
            while (received < threadPool && !errors) {
                try {
                    Future<Double> resultFuture = completionService.take();
                    Double result = resultFuture.get();
                    if (sse.get() == null) {
                        sse.set(result);
                    } else {
                        sse.set(result + sse.get());
                    }

                    received++;
                } catch (Exception e) {
                    e.printStackTrace();
                    errors = true;
                }
            }

            averageWeightsAndBiases(networks);
        }

        return sse.get();
    }

    private void averageWeightsAndBiases(NeuralNetwork[] networks) {
        double[][][] weightMatrix = new double[networks[0].hidden.size() + 1][][];
        double[][] biasMatrix = new double[networks[0].hidden.size() + 1][];

        for (NeuralNetwork network : networks) {
            for (int layer = 0; layer <= network.hidden.size(); layer++) {
                PerceptronLayer<LearningPerceptron> lpLayer;
                if (layer < network.hidden.size()) {
                    lpLayer = network.hidden.get(layer);
                } else {
                    lpLayer = output;
                }

                weightMatrix[layer] = new double[lpLayer.getSize()][];
                biasMatrix[layer] = new double[lpLayer.getSize()];
                for (int perceptronIndex = 0; perceptronIndex < lpLayer.getSize(); perceptronIndex++) {
                    LearningPerceptron cast = lpLayer.getPerceptron(perceptronIndex);
                    biasMatrix[layer][perceptronIndex] = cast.getBias() / (double) networks.length;
                    weightMatrix[layer][perceptronIndex] = new double[cast.getWeights().length];
                    for (int weightIndex = 0; weightIndex < cast.getWeights().length; weightIndex++) {
                        weightMatrix[layer][perceptronIndex][weightIndex] += cast.getWeights()[weightIndex] / (double) networks.length;
                    }
                }
            }
        }

        for (NeuralNetwork network : networks) {
            for (int layer = 0; layer <= network.hidden.size(); layer++) {
                PerceptronLayer<LearningPerceptron> lpLayer;
                if (layer < network.hidden.size()) {
                    lpLayer = network.hidden.get(layer);
                } else {
                    lpLayer = output;
                }

                weightMatrix[layer] = new double[lpLayer.getSize()][];
                for (int perceptronIndex = 0; perceptronIndex < lpLayer.getSize(); perceptronIndex++) {
                    LearningPerceptron cast = lpLayer.getPerceptron(perceptronIndex);
                    cast.setBias(biasMatrix[layer][perceptronIndex]);

                    weightMatrix[layer][perceptronIndex] = new double[cast.getWeights().length];
                    for (int weightIndex = 0; weightIndex < cast.getWeights().length; weightIndex++) {
                        cast.setWeight(weightIndex, weightMatrix[layer][perceptronIndex][weightIndex]);
                    }
                }
            }
        }
    }

    public double train(double[] inputs, double[] outputs) {
        double[][] result = computeEpochMatrix(inputs);
        double[] predictions = result[hidden.size() + 1];

        Map<PerceptronLayer, double[]> errorMap = backPropogatedErrorGradient(predictions, outputs, result);
        PerceptronLayer current = output;
        int layerIndex = hidden.size() + 1;
        /*
         * When training we want to update all weights going into every layer except the input layer,
         * so we loop until we encounter the input layer
         */

        while (!(current instanceof InputLayer)) {
            //This current layer's error values as computed using the back propogated error
            double[] errors = errorMap.get(current);
            /*
             * For each perceptron in this layer (we can assume they are learning perceptrons as the input layer
             * is where we stop
             */
            for (int index = 0; index < current.getSize(); index++) {
                LearningPerceptron lp = (LearningPerceptron) current.getPerceptron(index);
                // Derivative of sigmoid(E x_iw_j) is actually just result of perceptron instead of sigmoid

                for (int weightIndex = 0; weightIndex < lp.getInputLength(); weightIndex++) {
                    double diff = errors[index] * result[layerIndex - 1][weightIndex];
                    lp.updateWeight(weightIndex, diff);
                }

                lp.updateBias(errors[index]);
            }

            current = getPrevious(current);
            layerIndex--;
        }

        // Return the Sum Squared Error as a result when training
        predictions = computeEpochMatrix(inputs)[hidden.size() + 1];
        double sError = 0;
        for (int i = 0; i < outputs.length; i++) {
            sError += Math.pow((outputs[i] - predictions[i]), 2);
        }

        return sError;
    }


    private Map<PerceptronLayer, double[]> backPropogatedErrorGradient(double[] predictions, double[] expected, double[][] matrix) {
        // Create a map that stores all the errors for each layer
        Map<PerceptronLayer, double[]> gradientMap = new HashMap<>();

        double[] outputGradients = new double[predictions.length];
        for (int i = 0; i < predictions.length; i++) {
            double err_i = expected[i] - predictions[i];
            LearningPerceptron perceptron = (LearningPerceptron) output.getPerceptron(i);
            double weightedProductI = perceptron.getWeightedProduct(matrix[hidden.size()]);
            outputGradients[i] = err_i * output.getPerceptron(i).getActivation().derive(weightedProductI);
        }

        gradientMap.put(output, outputGradients);


        PerceptronLayer previousLayer = output;
        // Backwardly iterate over the hidden layers to feed back the error values
        for (int i = hidden.size() - 1; i >= 0; i--) {
            PerceptronLayer currentLayer = hidden.get(i);

            // Get the error values for the previous layer
            double[] prevLayerGradients = gradientMap.get(previousLayer);
            double[] layerGradients = new double[currentLayer.getSize()];

            // Loop over the perceptrons in the current layer
            for (int index = 0; index < currentLayer.getSize(); index++) {
                LearningPerceptron current = (LearningPerceptron) currentLayer.getPerceptron(index);
                double weightedProductI = current.getWeightedProduct(matrix[i]);
                double derivative = currentLayer.getPerceptron(index).getActivation().derive(weightedProductI);
                // Loop over the previous layer's error values
                for (int x = 0; x < prevLayerGradients.length; x++) {

                    // Get the perceptron in the previous layer this current perceptron is connected to
                    LearningPerceptron prevPerceptron = (LearningPerceptron) previousLayer.getPerceptron(x);

                    // Add the previous error for that perceptron multiplied by the weight between this and that perceptron
                    layerGradients[index] += prevLayerGradients[x] * derivative * prevPerceptron.getWeights()[index];
                }
            }

            // Put the errors in the map and move a layer backwards
            gradientMap.put(currentLayer, layerGradients);
            previousLayer = currentLayer;
        }

        return gradientMap;
    }


    public PerceptronLayer getPrevious(PerceptronLayer from) {
        if (from == output) {
            if (hidden.size() == 0) {
                return input;
            } else {
                return hidden.get(hidden.size() - 1);
            }
        }

        for (int i = hidden.size() - 1; i >= 0; i--) {
            PerceptronLayer current = hidden.get(i);
            if (current == from && i > 0) {
                return hidden.get(i - 1);
            }
        }

        return input;
    }

    public double[] activationPredict(double[] inputs, ActivationFunction av) {
        double[] results = computeEpochMatrix(inputs)[hidden.size() + 1];
        for (int i = 0; i < results.length; i++) {
            results[i] = av.activate(results[i]);
        }

        return results;
    }

    public double[] binaryPredict(double[] inputs) {
        double[] results = computeEpochMatrix(inputs)[hidden.size() + 1];
        for (int i = 0; i < results.length; i++) {
            results[i] = Math.round(results[i]);
        }

        return results;
    }

    public double[] predict(double[] inputs) {
        return computeEpochMatrix(inputs)[hidden.size() + 1];
    }

    // TESTED
    private double[][] computeEpochMatrix(double[] inputs) {
        double[][] result = new double[2 + hidden.size()][];
        result[0] = input.predict(inputs);

        for (int i = 0; i < hidden.size(); i++) {
            result[i + 1] = hidden.get(i).predict(result[i]);
        }

        result[hidden.size() + 1] = output.predict(result[hidden.size()]);
        return result;
    }

    public PerceptronLayer<InputPerceptron> getInputLayer() {
        return input;
    }

    public PerceptronLayer<LearningPerceptron> getOutputLayer() {
        return output;
    }

    public List<PerceptronLayer<LearningPerceptron>> getHiddenLayers() {
        return hidden;
    }

}
