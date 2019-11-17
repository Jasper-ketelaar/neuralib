package nl.yasper.neuralib.display;

import nl.yasper.neuralib.network.NeuralNetwork;

import javax.swing.*;
import java.util.concurrent.Future;
import java.util.concurrent.FutureTask;

public class NetworkFrame extends JFrame {

    private final NeuralNetwork network;
    private final NetworkPanel panel;

    public NetworkFrame(NeuralNetwork network) {
        super("Network Display");

        this.network = network;
        this.panel = new NetworkPanel(network);

        setLocationRelativeTo(null);
        setContentPane(panel);
    }

    public static Future<NetworkFrame> display(NeuralNetwork network) {
        FutureTask<NetworkFrame> futureTask = new FutureTask<>(() -> {
            NetworkFrame frame = new NetworkFrame(network);
            frame.pack();
            frame.validate();
            frame.setVisible(true);
            return frame;
        });

        SwingUtilities.invokeLater(futureTask);

        return futureTask;
    }

    public NetworkPanel getPanel() {
        return panel;
    }
}
