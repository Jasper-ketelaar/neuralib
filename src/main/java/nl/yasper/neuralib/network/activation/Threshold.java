package nl.yasper.neuralib.network.activation;

public class Threshold implements ActivationFunction {

    private final double value;

    public Threshold(double value) {
        this.value = value;
    }

    @Override
    public double activate(double output) {
        if (output > value) {
            return  1;
        }

        return  0;
    }

    @Override
    public double derive(double output) {
        return output;
    }
}