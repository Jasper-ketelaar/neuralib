package nl.yasper.neuralib.osrs.struct.record;


import java.awt.*;
import java.util.ArrayList;

public class MousePathData {

    private final ArrayList<MousePathEntry> entries = new ArrayList<>();
    private long endTime;
    private boolean drag;
    private boolean human;

    public MousePathData(long time, boolean drag) {
        this.endTime = time;
        this.drag = drag;
    }

    public MousePathData() {
    }

    public void addPoint(Point point) {
        this.entries.add(new MousePathEntry(point, System.currentTimeMillis()));
        this.endTime = System.currentTimeMillis();
    }

    public boolean isHuman() {
        return human;
    }

    public void setHuman(boolean human){
        this.human = human;
    }

    public MousePathEntry getStartEntry() {
        return entries.get(0);
    }

    public MousePathEntry getEndEntry() {
        return entries.get(entries.size() - 1);
    }

    public Point getStartPoint() {
        return entries.get(0).getPoint();
    }

    public int getStartY() {
        return getStartPoint().y;
    }

    public int getStartX() {
        return getStartPoint().x;
    }

    public Point getEndPoint() {
        return entries.get(entries.size() - 1).getPoint();
    }

    public int getEndX() {
        return getEndPoint().x;
    }

    public int getEndY() {
        return getEndPoint().y;
    }

    public double getTrueDistance() {
        return new Point(getStartX(), getStartY()).distance(getEndX(), getEndY());
    }

    public ArrayList<MousePathEntry> getEntries() {
        return entries;
    }

    public int getPathDistance() {
        int dist = 0;
        MousePathEntry prev = entries.get(0);
        for (MousePathEntry entry : entries) {
            dist += prev.getPoint().distance(entry.getPoint());
            prev = entry;
        }

        return dist;
    }

    public long getEndTime() {
        return endTime;
    }

    public boolean isDrag() {
        return drag;
    }
}
