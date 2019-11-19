package nl.yasper.neuralib.network.layer;

import nl.yasper.neuralib.network.perceptron.LearningPerceptron;

public class SoftMaxOutputLayer extends PerceptronLayer<LearningPerceptron> {

    public SoftMaxOutputLayer(int size) {
        super(size);
    }

    @Override
    public double[] predict(double[] input) {
        double[] result = new double[getSize()];
        for (int i = 0; i < result.length; i++) {
            result[i] = getPerceptron(i).getWeightedProduct(input);
        }

        double max = Double.MIN_VALUE;
        for (double v : result) {
            max = Math.max(max, v);
        }

        for (int i = 0; i < result.length; i++) {
            result[i] = result[i] - max;
        }

        double denominator = 0;
        for (double v : result) {
            denominator += Math.exp(v);
        }

        for (int i = 0; i < result.length; i++) {
            result[i] = Math.max(0, Math.exp(result[i]) / denominator);
            if (Double.isNaN(result[i])) {
                result[i] = 0;
            }
        }

        return result;
    }
}
