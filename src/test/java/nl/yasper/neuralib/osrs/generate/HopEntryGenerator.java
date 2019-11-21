package nl.yasper.neuralib.osrs.generate;

import nl.yasper.neuralib.osrs.struct.record.MousePathData;
import nl.yasper.neuralib.osrs.struct.record.MousePathEntry;

import java.awt.*;

public class HopEntryGenerator implements MouseEntryGenerator {


    @Override
    public MousePathData generateBetween(Point p1, Point p2, long start, long end) {
        MousePathData bot = new MousePathData();
        bot.getEntries().add(new MousePathEntry(p1, 0));
        bot.getEntries().add(new MousePathEntry(p2, 100));

        return bot;
    }
}
