package nl.yasper.neuralib.display;

import java.awt.*;

public interface Displayable {

    Dimension render(Graphics2D graphics);

    void setAbsolutePosition(Point position);

    Point getAbsolutePosition();

}
