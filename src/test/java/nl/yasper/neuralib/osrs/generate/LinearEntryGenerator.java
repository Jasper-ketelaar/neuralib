package nl.yasper.neuralib.osrs.generate;

import nl.yasper.neuralib.osrs.struct.record.MousePathData;
import nl.yasper.neuralib.osrs.struct.record.MousePathEntry;

import java.awt.*;

public class LinearEntryGenerator implements MouseEntryGenerator {


    @Override
    public MousePathData generateBetween(Point from, Point to, long start, long end) {
        MousePathData bot = new MousePathData();
        double dist = from.distance(to);
        long time = (start - end);
        long dt = (long) (time / dist);
        double dx = (to.getX() - from.getX()) / dist;
        double dy = (to.getY() - from.getY()) / dist;

        MousePathEntry current = new MousePathEntry(from, 0);
        double xDiffAcc = 0.0;
        double yDiffAcc = 0.0;
        int counter = 0;
        while (current.getPoint().distance(to) > 4.0 && counter++ < dist) {
            bot.getEntries().add(current);
            xDiffAcc += dx;
            yDiffAcc += dy;

            double newX = current.getPoint().getX();
            double newY = current.getPoint().getY();

            if (Math.abs(xDiffAcc) >= 1.0) {
                newX += xDiffAcc;
                xDiffAcc = 0;
            }

            if (Math.abs(yDiffAcc) >= 1.0) {
                newY += yDiffAcc;
                yDiffAcc = 0;
            }

            current = new MousePathEntry(new Point((int) newX, (int) newY), current.getTime() + dt);
        }

        return bot;
    }

}
