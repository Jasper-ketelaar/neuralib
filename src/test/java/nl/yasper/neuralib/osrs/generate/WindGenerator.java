package nl.yasper.neuralib.osrs.generate;

import nl.yasper.neuralib.osrs.struct.record.MousePathData;
import nl.yasper.neuralib.osrs.struct.record.MousePathEntry;

import java.awt.*;
import java.util.ArrayList;

public class WindGenerator implements MouseEntryGenerator {

    private synchronized ArrayList<MousePathEntry> windMouseImpl(double xs, double ys, double xe, double ye, double gravity, double wind, double minWait, double maxWait, double maxStep, double targetArea) {
        final double sqrt3 = Math.sqrt(3);
        final double sqrt5 = Math.sqrt(5);
        final ArrayList<MousePathEntry> mpes = new ArrayList<>();

        double dist, veloX = 0, veloY = 0, windX = 0, windY = 0;
        Point current = new Point((int) xs, (int) ys);
        while ((dist = Math.hypot(xs - xe, ys - ye)) >= 1) {
            wind = Math.min(wind, dist);
            if (dist >= targetArea) {
                windX = windX / sqrt3 + (2D * Math.random() - 1D) * wind / sqrt5;
                windY = windY / sqrt3 + (2D * Math.random() - 1D) * wind / sqrt5;
            } else {
                windX /= sqrt3;
                windY /= sqrt3;
                if (maxStep < 3) {
                    maxStep = Math.random() * 3D + 3D;
                } else {
                    maxStep /= sqrt5;
                }
                //System.out.println(maxStep + ":" + windX + ";" + windY);
            }
            veloX += windX + gravity * (xe - xs) / dist;
            veloY += windY + gravity * (ye - ys) / dist;
            double veloMag = Math.hypot(veloX, veloY);
            if (veloMag > maxStep) {
                double randomDist = maxStep / 2D + Math.random() * maxStep / 2D;
                veloX = (veloX / veloMag) * randomDist;
                veloY = (veloY / veloMag) * randomDist;
            }
            xs += veloX;
            ys += veloY;
            int mx = (int) Math.round(xs);
            int my = (int) Math.round(ys);
            if (current.x != mx || current.y != my) {
                double step = Math.hypot(xs - current.x, ys - current.y);
                mpes.add(new MousePathEntry(new Point(mx, my), (long) ((maxWait - minWait) * (step / maxStep) + minWait)));
                current = new Point(mx, my);
            }

        }

        return mpes;
    }


    @Override
    public MousePathData generateBetween(Point p1, Point p2, long start, long end) {
        MousePathData mpd = new MousePathData();
        double speed = p1.distance(p2) / (end - start);
        ArrayList<MousePathEntry> wind = windMouseImpl(p1.getX(), p1.getY(), p2.getX(), p2.getY(), 9D, 3D, 5D / speed, 10D / speed, 10D * speed, 8D * speed);
        mpd.getEntries().addAll(wind);
        return mpd;
    }
}
