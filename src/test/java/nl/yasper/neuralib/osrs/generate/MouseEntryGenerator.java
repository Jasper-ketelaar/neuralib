package nl.yasper.neuralib.osrs.generate;

import nl.yasper.neuralib.osrs.struct.record.MousePathData;
import nl.yasper.neuralib.osrs.struct.record.MousePathEntry;

import java.awt.*;

public interface MouseEntryGenerator {

    default MousePathData generate(MousePathData human) {
        MousePathEntry start = human.getStartEntry();
        MousePathEntry end = human.getEndEntry();

        return generateBetween(start.getPoint(), end.getPoint(), start.getTime(), end.getTime());
    }

    MousePathData generateBetween(Point p1, Point p2, long start, long end);

}
