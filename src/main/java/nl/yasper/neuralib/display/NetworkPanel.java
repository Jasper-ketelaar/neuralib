package nl.yasper.neuralib.display;

import nl.yasper.neuralib.network.NeuralNetwork;
import nl.yasper.neuralib.network.layer.InputLayer;
import nl.yasper.neuralib.network.layer.PerceptronLayer;

import javax.swing.*;
import java.awt.*;

public class NetworkPanel extends JPanel {

    private final NeuralNetwork network;

    public NetworkPanel(NeuralNetwork network) {
        setPreferredSize(new Dimension(600, 300));
        this.network = network;
    }

    @Override
    public void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g;
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING,
                RenderingHints.VALUE_ANTIALIAS_ON);

        g.drawLine(0, 150, 600, 150);

        int index = 0;
        drawLayer(g, network.getInputLayer(), index++);
        for (PerceptronLayer layer : network.getHiddenLayers()) {
            drawLayer(g, layer, index++);
        }
        drawLayer(g, network.getOutputLayer(), index);
    }

    private void drawLayer(Graphics g, PerceptronLayer layer, int layerIndex) {
        int startY = 10;
        int startX = 50 + layerIndex * 150;
        for (int i = 0; i < layer.getSize(); i++) {
            if (layer instanceof InputLayer) {
                g.setColor(Color.GREEN);
            } else if (layerIndex == network.getHiddenLayers().size() + 1) {
                g.setColor(Color.BLUE);
            } else {
                g.setColor(Color.WHITE);
            }
            g.fillOval(startX, startY + i * 100, 50, 50);
            g.setColor(Color.BLACK);
            g.drawOval(startX, startY + i * 100, 50, 50);
        }
    }

}
