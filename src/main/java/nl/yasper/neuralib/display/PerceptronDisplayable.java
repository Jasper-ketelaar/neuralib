package nl.yasper.neuralib.display;

import nl.yasper.neuralib.network.perceptron.InputPerceptron;
import nl.yasper.neuralib.network.perceptron.Perceptron;

import java.awt.*;

public class PerceptronDisplayable implements Displayable {

    protected static final int PERCEPTRON_SIZE = 35;

    private final Perceptron perceptron;
    private final int index;
    private Point position;
    private boolean hidden;


    public PerceptronDisplayable(Perceptron perceptron, int index, boolean hidden) {
        this.perceptron = perceptron;
        this.index = index;
        this.hidden = hidden;
    }

    @Override
    public Dimension render(Graphics2D graphics) {
        graphics.translate(0, PERCEPTRON_SIZE * 2);

        String text = "";
        if (perceptron instanceof InputPerceptron) {
            text = String.format("x_%d", index + 1);
        } else if (hidden) {
            text = String.format("h_%d", index + 1);
        } else {
            text = String.format("o_%d", index + 1);
        }

        Font font = graphics.getFont();
        graphics.setFont(font.deriveFont(16f));
        FontMetrics fm = graphics.getFontMetrics();
        int textLength = fm.stringWidth(text);
        int textHeight = fm.getHeight();

        graphics.drawOval(0, 0, PERCEPTRON_SIZE, PERCEPTRON_SIZE);

        graphics.drawString(
                text,
                PERCEPTRON_SIZE / 2 - textLength / 2,
                PERCEPTRON_SIZE / 2 + (textHeight / 3)
        );

        return new Dimension(PERCEPTRON_SIZE, PERCEPTRON_SIZE * 2);
    }

    public Point getEndPosition() {
        Point clone = new Point(position);
        clone.translate(PERCEPTRON_SIZE / 2, -PERCEPTRON_SIZE * 3 / 2);
        return clone;
    }

    public Point getBeginPosition() {
        Point clone = new Point(position);
        clone.translate(-PERCEPTRON_SIZE / 2 - 2, -PERCEPTRON_SIZE * 3 / 2);
        return clone;
    }

    @Override
    public Point getAbsolutePosition() {
        return position;
    }

    @Override
    public void setAbsolutePosition(Point position) {
        this.position = position;
    }

    public int getIndex() {
        return index;
    }

}
