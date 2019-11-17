package nl.yasper.neuralib.display.displayable;

import nl.yasper.neuralib.network.layer.PerceptronLayer;
import nl.yasper.neuralib.network.perceptron.LearningPerceptron;

import java.awt.*;
import java.awt.geom.AffineTransform;
import java.util.ArrayList;

public class LayerDisplayable implements Displayable {

    private final PerceptronLayer layer;
    private final ArrayList<PerceptronDisplayable> displayablePerceptrons;
    private final boolean hidden;

    private Point position = new Point(0, 0);
    private int perceptronPosition;

    public LayerDisplayable(PerceptronLayer layer, boolean hidden) {
        this.layer = layer;
        this.hidden = hidden;
        this.displayablePerceptrons = new ArrayList<>(layer.getSize());
        for (int i = 0; i < layer.getSize(); i++) {
            this.displayablePerceptrons.add(new PerceptronDisplayable(layer.getPerceptron(i), i, hidden));
        }
    }

    private Dimension drawPerceptron(Graphics2D g, int index) {
        PerceptronDisplayable pd = displayablePerceptrons.get(index);
        Dimension perceptronDimension = pd.render(g);
        perceptronPosition += perceptronDimension.height * 2;
        pd.setAbsolutePosition(new Point(position.x, perceptronPosition));
        g.translate(0, perceptronDimension.height);
        return perceptronDimension;
    }

    @Override
    public Dimension render(Graphics2D graphics) {
        Dimension total = new Dimension(50, 0);
        for (int i = 0; i < layer.getSize(); i++) {
            Dimension perc = drawPerceptron(graphics, i);
            total = new Dimension(total.width + perc.width, total.height + perc.height);
        }

        total = new Dimension(total.width, total.height);

        perceptronPosition = 0;
        return total;
    }

    @Override
    public Point getAbsolutePosition() {
        return position;
    }

    @Override
    public void setAbsolutePosition(Point position) {
        this.position = position;
    }

    public void connect(Graphics2D g, LayerDisplayable to) {
        for (PerceptronDisplayable pd : this.displayablePerceptrons) {
            for (PerceptronDisplayable pd2 : to.displayablePerceptrons) {
                connect(g, pd, pd2);
            }
        }
    }

    public void connect(Graphics2D g, PerceptronDisplayable first, PerceptronDisplayable second) {
        if (first.getAbsolutePosition() == null) {
            return;
        }

        Point firstPoint = first.getEndPosition();
        Point secondPoint = second.getBeginPosition();
        g.drawLine(firstPoint.x, firstPoint.y, secondPoint.x, secondPoint.y);

        AffineTransform orig = g.getTransform();
        int deltaX = secondPoint.x - firstPoint.x;
        int deltaY = secondPoint.y - firstPoint.y;

        Point halfway = new Point(firstPoint.x + deltaX / 2, firstPoint.y + deltaY / 2);
        double theta = Math.atan2(deltaX, deltaY);
        g.rotate(-theta + Math.PI / 2, halfway.x, halfway.y);


        double weight = ((LearningPerceptron) second.getPerceptron()).getWeights()[first.getIndex()];
        g.drawString(String.format("w%d_%d: %.4f", first.getIndex() + 1, second.getIndex() + 1, weight), halfway.x, halfway.y - 5);
        g.setTransform(orig);
    }
}
