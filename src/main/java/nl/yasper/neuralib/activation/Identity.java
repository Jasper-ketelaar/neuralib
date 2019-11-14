package nl.yasper.neuralib.activation;

public class Identity implements ActivationFunction {

    @Override
    public double activate(double output) {
        return 0;
    }

    @Override
    public double derive(double output) {
        return 0;
    }
}