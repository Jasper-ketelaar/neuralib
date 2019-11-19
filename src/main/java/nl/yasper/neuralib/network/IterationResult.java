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

    public IterationResult add(IterationResult other) {
        double[][][] newUpdates = new double[updates.length][][];
        for (int x = 0; x < updates.length; x++) {
            newUpdates[x] = new double[updates[x].length][];
            for (int y = 0; y < updates[x].length; y++) {
                newUpdates[x][y] = new double[updates[x][y].length];
                for (int z = 0; z < updates[x][y].length; z++) {
                    newUpdates[x][y][z] = updates[x][y][z] + other.updates[x][y][z];
                }
            }
        }

        return new IterationResult(newUpdates, getSSE() + other.getSSE());
    }

}
