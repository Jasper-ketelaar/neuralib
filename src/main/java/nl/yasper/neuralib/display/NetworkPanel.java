package nl.yasper.neuralib.display;

import nl.yasper.neuralib.network.NeuralNetwork;

import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;

public class NetworkPanel extends JPanel implements Runnable {

    private final NeuralNetwork network;
    private final ArrayList<LayerDisplayable> layers;

    private Dimension dimensions = new Dimension(600, 300);
    private int layerBaseX;

    public NetworkPanel(NeuralNetwork network) {
        setPreferredSize(dimensions);
        this.network = network;
        this.layers = new ArrayList<>(2 + network.getHiddenLayers().size());
        this.layers.add(new LayerDisplayable(network.getInputLayer(), false));
        network.getHiddenLayers().forEach(hl -> layers.add(new LayerDisplayable(hl, true)));
        this.layers.add(new LayerDisplayable(network.getOutputLayer(), false));

        startRenderLoop();
    }

    private void startRenderLoop() {
        new Thread(this).start();
    }

    @Override
    public void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g;

        layerBaseX += 20;
        g2.translate(layerBaseX, 0);
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);


        LayerDisplayable previous = layers.get(0);
        for (LayerDisplayable ld : layers) {
            if (ld == previous) {
                continue;
            }

            previous.connect(g2, ld);
            previous = ld;
        }

        int wd = 0;
        for (LayerDisplayable ld : layers) {
            wd += renderLayer(g2, ld);
        }

        if (wd > dimensions.width) {
            dimensions = new Dimension(wd, dimensions.height);
        }

        layerBaseX = 0;
    }

    private int renderLayer(Graphics2D g2, LayerDisplayable layer) {
        Dimension rendered = layer.render(g2);
        g2.translate(rendered.width * 2, -rendered.height * 2);
        layer.setAbsolutePosition(new Point(layerBaseX, 0));
        layerBaseX += rendered.width * 2;

        int newHeight = (rendered.height * 2 + 2 * PerceptronDisplayable.PERCEPTRON_SIZE);
        if (newHeight > dimensions.height) {
            dimensions = new Dimension(dimensions.width, newHeight);
        }

        return rendered.width * 2;
    }


    @Override
    public void run() {
        while (isVisible()) {
            if (!getPreferredSize().equals(dimensions)) {
                setPreferredSize(dimensions);
                Component parent = getParent().getParent().getParent();
                if (parent instanceof JFrame) {
                    parent.setSize(dimensions);
                }
            }
            repaint();
            try {
                Thread.sleep(50);
            } catch (InterruptedException ignored) {
            }
        }
    }
}
