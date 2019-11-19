package nl.yasper.neuralib.network;

public class IterationResult {

    private final double[][][] updates;
    private final double sse;

    public IterationResult(double[][][] updates, double sse) {
        this.updates = updates;
        this.sse = sse;
    }

    public double[][][] getUpdates() {
        return updates;
    }

    public double getSSE() {
        return sse;
    }

}
