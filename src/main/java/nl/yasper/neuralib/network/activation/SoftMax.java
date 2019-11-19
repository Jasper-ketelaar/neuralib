package nl.yasper.neuralib.network.activation;

public class SoftMax implements ActivationFunction {

    private double denominator;

    public void setDenominator(double denominator) {
        this.denominator = denominator;
    }

    @Override
    public double activate(double output) {
        double res = Math.exp(output) / denominator;
        if (Double.isNaN(res)) {
            return 0;
        }

        return res;
    }

    @Override
    public double derive(double output) {
        return 1;
    }
}
