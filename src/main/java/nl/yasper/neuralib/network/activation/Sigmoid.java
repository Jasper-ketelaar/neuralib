package nl.yasper.neuralib.network.activation;

public class Sigmoid implements ActivationFunction {

    @Override
    public double activate(double output) {
        return 1.0 / (1.0 + Math.pow(Math.E, -output));
    }

    @Override
    public double derive(double output) {
        return 0;
    }

}
