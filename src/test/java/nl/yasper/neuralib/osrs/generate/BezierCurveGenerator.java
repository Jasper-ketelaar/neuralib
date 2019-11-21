package nl.yasper.neuralib.osrs.generate;

import nl.yasper.neuralib.osrs.struct.record.MousePathData;
import nl.yasper.neuralib.osrs.struct.record.MousePathEntry;

import java.awt.*;

public class BezierCurveGenerator implements MouseEntryGenerator {

    @Override
    public MousePathData generateBetween(Point origin, Point point, long start, long end) {
        MousePathData data = new MousePathData();

        Point mid = new Point((origin.x + point.x) / 2, (origin.y + point.y) / 2);

        double distance = origin.distance(point);
        double speed = distance / (end - start);

        int bezierMidPointX = origin.x;
        int bezierMidPointY = mid.y;

        long prevTime = 0;
        for (double t = 0.0; t <= 1; t += 1 / distance) {
            int x = (int) ((1 - t) * (1 - t) * origin.x + 2 * (1 - t) * t * bezierMidPointX + t * t * point.x);
            int y = (int) ((1 - t) * (1 - t) * origin.y + 2 * (1 - t) * t * bezierMidPointY + t * t * point.y);

            long time = (long) (distance * speed * t);
            MousePathEntry mpe = new MousePathEntry(new Point(x, y), prevTime + time);
            prevTime += time;

            data.getEntries().add(mpe);
        }


        return data;
    }

}
