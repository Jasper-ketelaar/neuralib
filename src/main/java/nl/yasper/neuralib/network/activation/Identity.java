package nl.yasper.neuralib.network.activation;

public class Identity implements ActivationFunction {

    @Override
    public double activate(double output) {
        return output;
    }

    @Override
    public double derive(double output) {
        return 0;
    }
}