package nl.yasper.neuralib.network.activation;

public class ReLU implements ActivationFunction {

    @Override
    public double activate(double output) {
        return Math.max(0, output);
    }

    @Override
    public double derive(double output) {
        if (output > 0) {
            return 1;
        } else {
            return 0;
        }
    }
}
