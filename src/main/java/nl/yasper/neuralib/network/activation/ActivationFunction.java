package nl.yasper.neuralib.network.activation;

public interface ActivationFunction {

    ActivationFunction IDENTITY = new Identity();
    ActivationFunction SIGMOID = new Sigmoid();
    ActivationFunction RELU = new ReLU();
    ActivationFunction SOFTMAX = new SoftMax();

    double activate(double output);

    double derive(double output);

}