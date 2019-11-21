package nl.yasper.neuralib.osrs.struct.flat;

import java.util.List;

public class FlattenedMouseEntry {

    private double x;
    private double y;
    private double velocity;

    public FlattenedMouseEntry(double x, double y, double velocity) {
        this.x = x;
        this.y = y;
        this.velocity = velocity;
    }

    public static double[] flatten(List<FlattenedMouseEntry> entries) {
        double[] result = new double[120];
        for (int i = 0; i < entries.size(); i++) {
            FlattenedMouseEntry entry = entries.get(i);
            result[i] = entry.x;
            result[i + 1] = entry.y;
            result[i + 2] = entry.velocity;
        }

        return result;
    }

    public double getX() {
        return x;
    }

    public double getY() {
        return y;
    }

    public double getVelocity() {
        return velocity;
    }

    public void setX(double x) {
        this.x = x;
    }

    public void setY(double y) {
        this.y = y;
    }

    public void setVelocity(double velocity) {
        this.velocity = velocity;
    }

}
