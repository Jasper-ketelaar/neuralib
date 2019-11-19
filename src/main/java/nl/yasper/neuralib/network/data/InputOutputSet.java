package nl.yasper.neuralib.network.data;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;
import java.util.function.ToDoubleFunction;

public class InputOutputSet {

    private final double[][] inputs;
    private final double[][] outputs;

    public InputOutputSet(double[][] inputs, double[][] outputs) {
        this.inputs = inputs;
        this.outputs = outputs;
    }

    public static InputOutputSet csvToClassifier(InputStream stream, ToDoubleFunction<Double> normalizer, int start, int outputs) throws IOException {
        BufferedReader csvReader = new BufferedReader(new InputStreamReader(stream));
        int ctr = 0;
        while (ctr++ < start) {
            csvReader.readLine();
        }

        String line;

        Map<double[], double[]> csvMap = new HashMap<>();
        while ((line = csvReader.readLine()) != null) {
            String[] split = line.split(",");
            double[] result = new double[split.length];
            double[] input = new double[split.length - 1];
            for (int i = 1; i < split.length; i++) {
                input[i - 1] = normalizer.applyAsDouble(Double.parseDouble(split[i]));
            }

            double[] output = new double[outputs];
            output[Integer.parseInt(split[0])] = 1;
            csvMap.put(input, output);
        }

        return new InputOutputSet(csvMap.keySet().toArray(new double[0][]), csvMap.values().toArray(new double[0][]));
    }

    public double[][] getInputs() {
        return inputs;
    }

    public double[][] getOutputs() {
        return outputs;
    }

    public InputOutputSet[] split(int atIndex) {
        InputOutputSet[] result = new InputOutputSet[2];

        double[][] beforeInputs = new double[atIndex][];
        System.arraycopy(inputs, 0, beforeInputs, 0, atIndex);
        double[][] beforeOutputs = new double[atIndex][];
        System.arraycopy(outputs, 0, beforeOutputs, 0, atIndex);
        result[0] = new InputOutputSet(beforeInputs, beforeOutputs);

        int remaining = inputs.length - atIndex;
        double[][] afterInputs = new double[remaining][];
        System.arraycopy(inputs, atIndex, afterInputs, 0, remaining);
        double[][] afterOutputs = new double[remaining][];
        System.arraycopy(outputs, atIndex, afterOutputs, 0, remaining);
        result[1] = new InputOutputSet(afterInputs, afterOutputs);


        return result;
    }
}
