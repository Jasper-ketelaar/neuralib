package nl.yasper.neuralib.osrs.struct.record;

import java.awt.*;

public class MousePathEntry {

    private final Point point;
    private final long time;

    public MousePathEntry(Point point, long time) {
        this.point = point;
        this.time = time;
    }

    public Point getPoint() {
        return point;
    }

    public long getTime() {
        return time;
    }

}
