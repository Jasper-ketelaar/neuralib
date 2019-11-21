package nl.yasper.neuralib.osrs;

import com.google.gson.*;
import nl.yasper.neuralib.PrintAndStoreStream;
import nl.yasper.neuralib.display.NetworkFrame;
import nl.yasper.neuralib.display.NetworkPanel;
import nl.yasper.neuralib.network.NeuralNetwork;
import nl.yasper.neuralib.network.activation.ActivationFunction;
import nl.yasper.neuralib.network.builder.NetworkBuilder;
import nl.yasper.neuralib.network.data.InputOutputSet;
import nl.yasper.neuralib.osrs.generate.*;
import nl.yasper.neuralib.osrs.struct.flat.FlattenedMouseEntry;
import nl.yasper.neuralib.osrs.struct.flat.FlattenedMousePath;
import nl.yasper.neuralib.osrs.struct.record.MousePathData;
import nl.yasper.neuralib.osrs.struct.record.MousePathEntry;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.awt.*;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.*;

public class TestRSMouseData {

    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
    private static final MouseEntryGenerator[] GENERATORS = {new WindGenerator(), new HopEntryGenerator(), new LinearEntryGenerator(), new BezierCurveGenerator()};

    private List<MousePathData> getPathList() throws IOException {
        Path resources = Paths.get("src", "test", "resources", "nl", "yasper", "neuralib", "recorded-data");
        File directory = resources.toFile();
        if (!directory.exists() || !directory.isDirectory()) {
            Assert.fail("Resource directory doesn't exist");
        }

        List<MousePathData> pathJsonList = new ArrayList<>();
        for (File file : Objects.requireNonNull(directory.listFiles((file, name) -> name.endsWith(".act")))) {
            JsonObject parsed = GSON.fromJson(new FileReader(file), JsonObject.class);
            JsonArray data = parsed.getAsJsonArray("data");
            for (JsonElement entry : data) {
                JsonObject dataEntry = entry.getAsJsonObject();
                if (!dataEntry.get("className").getAsString().equals("MousePathData")) {
                    continue;
                }

                MousePathData mpd = GSON.fromJson(dataEntry, MousePathData.class);
                mpd.setHuman(true);

                if (mpd.getEntries().size() >= 40) {
                    pathJsonList.add(mpd);
                }
            }
        }

        return pathJsonList;
    }

    private List<MousePathData> padWithGeneration(List<MousePathData> paths) {
        List<MousePathData> paddedList = new ArrayList<>(paths.size() * 2);
        Random random = new Random(1248124719);
        for (MousePathData mpd : paths) {
            paddedList.add(mpd);
            int randomGenIndex = random.nextInt(GENERATORS.length);
            MousePathData padded = GENERATORS[randomGenIndex].generate(mpd);
            padded.setHuman(false);
            paddedList.add(padded);
        }

        return paddedList;
    }

    private List<MousePathData> convertRsPacketFormat(List<MousePathData> original) {
        List<MousePathData> pathsRsPacket = new ArrayList<>();
        for (MousePathData path : original) {
            pathsRsPacket.addAll(convertRsPacketFormat(path));
        }

        return pathsRsPacket;
    }

    private List<MousePathData> convertRsPacketFormat(MousePathData original) {
        List<MousePathData> pathsRsPacket = new ArrayList<>();
        long last = 0;
        for (int i = 0; i < original.getEntries().size(); i++) {
            MousePathData rsFormat = new MousePathData();
            rsFormat.setHuman(original.isHuman());
            rsFormat.getEntries().add(original.getEntries().get(i));
            int entries = 1;
            for (int y = i; entries < 40 && y + entries < original.getEntries().size(); y++) {
                //System.out.println(i + " " + y);
                MousePathEntry pathEntry = original.getEntries().get(y + entries);
                long time = pathEntry.getTime();
                if (time - last < 50) {
                    continue;
                }

                entries++;
                y--;
                rsFormat.getEntries().add(pathEntry);
                last = time;
            }

            if (rsFormat.getEntries().size() > 20) {
                pathsRsPacket.add(rsFormat);
            }
        }
        return pathsRsPacket;
    }

    private List<FlattenedMousePath> flattenAndMap(List<MousePathData> rsPacketList) {
        List<FlattenedMousePath> flattenedMouseEntries = new ArrayList<>();
        for (MousePathData mpd : rsPacketList) {
            long time = 0;
            Point point = null;
            FlattenedMouseEntry[] flattenedArray = new FlattenedMouseEntry[mpd.getEntries().size()];

            for (int i = 0; i < mpd.getEntries().size(); i++) {
                MousePathEntry entry = mpd.getEntries().get(i);
                if (point == null) {
                    point = entry.getPoint();
                    time = entry.getTime();
                }

                double velocity = Math.abs(point.distance(entry.getPoint()) / (time - entry.getTime()));
                if (Double.isNaN(velocity) || Double.isInfinite(velocity)) {
                    velocity = 0;
                }

                flattenedArray[i] = new FlattenedMouseEntry(entry.getPoint().getX(), entry.getPoint().getY(), velocity);

                point = entry.getPoint();
                time = entry.getTime();
            }

            flattenedMouseEntries.add(new FlattenedMousePath(flattenedArray, mpd.isHuman()));
        }

        return flattenedMouseEntries;
    }

    private void normalizeData(List<FlattenedMousePath> flattened) {
        int maxX = Integer.MIN_VALUE;
        int maxY = Integer.MIN_VALUE;
        double maxVel = Double.MIN_VALUE;

        for (FlattenedMousePath path : flattened) {
            for (FlattenedMouseEntry entry : path.getEntries()) {
                maxX = Math.max(maxX, (int) entry.getX());
                maxY = Math.max(maxY, (int) entry.getY());
                maxVel = Math.max(maxVel, entry.getVelocity());
            }
        }

        for (FlattenedMousePath path : flattened) {
            for (FlattenedMouseEntry entry : path.getEntries()) {
                entry.setX(entry.getX() / (double) maxX);
                entry.setY(entry.getY() / (double) maxY);
                entry.setVelocity(entry.getVelocity() / maxVel);
            }
        }
    }

    private FlattenedMousePath normalizeData(List<FlattenedMousePath> flattened, FlattenedMousePath toNormalize) {
        int maxX = Integer.MIN_VALUE;
        int maxY = Integer.MIN_VALUE;
        double maxVel = Double.MIN_VALUE;

        for (FlattenedMousePath path : flattened) {
            for (FlattenedMouseEntry entry : path.getEntries()) {
                maxX = Math.max(maxX, (int) entry.getX());
                maxY = Math.max(maxY, (int) entry.getY());
                maxVel = Math.max(maxVel, entry.getVelocity());
            }
        }

        for (FlattenedMouseEntry entry : toNormalize.getEntries()) {
            entry.setX(entry.getX() / (double) maxX);
            entry.setY(entry.getY() / (double) maxY);
            entry.setVelocity(entry.getVelocity() / maxVel);
        }

        return toNormalize;
    }


    private void accumulate() throws IOException {
        System.out.println("Accumulating all .act files");

        List<MousePathData> pathList = getPathList();
        List<MousePathData> generatedList = padWithGeneration(pathList);
        List<MousePathData> pathsRsPacket = convertRsPacketFormat(generatedList);
        List<FlattenedMousePath> flattened = flattenAndMap(pathsRsPacket);
        normalizeData(flattened);


        File paths = new File("./paths.json");
        if (!paths.exists() && !paths.createNewFile()) {
            Assert.fail("Cannot create file");
        }

        FileWriter fw = new FileWriter(paths);
        GSON.toJson(flattened, fw);
        fw.close();

        System.out.println("Normalized all data, padded it with generated data and wrote to paths.json");
    }

    private Map<double[], Double> get(boolean refresh) throws IOException {
        if (refresh) {
            accumulate();
        }

        File paths = new File("./paths.json");
        JsonArray pathArray = GSON.fromJson(new FileReader(paths), JsonArray.class);

        Map<double[], Double> flattenedList = new HashMap<>();
        for (JsonElement element : pathArray) {
            JsonObject pathObject = element.getAsJsonObject();
            List<FlattenedMouseEntry> entries = new ArrayList<>();
            for (JsonElement entry : pathObject.get(pathObject.keySet().iterator().next()).getAsJsonArray()) {
                JsonObject obj = entry.getAsJsonObject();
                FlattenedMouseEntry me = new FlattenedMouseEntry(obj.get("x").getAsDouble(), obj.get("y").getAsDouble(), obj.get("velocity").getAsDouble());
                entries.add(me);
            }

            flattenedList.put(FlattenedMouseEntry.flatten(entries), pathObject.get("human").getAsBoolean() ? 0.0 : 1.0);
        }

        return flattenedList;
    }

    @Test
    public void trainHumanNetwork() throws IOException {
        PrintAndStoreStream pos = new PrintAndStoreStream(new File("./training.log"));

        Map<double[], Double> data = get(false);
        double[][] entries = data.keySet().toArray(new double[0][]);
        double[][] outputs = data.values().stream().map(dbl -> new double[]{dbl}).toArray(double[][]::new);
        for (int i = 0; i < entries.length; i++) {
            outputs[i] = new double[]{data.get(entries[i])};
        }

        InputOutputSet ios = new InputOutputSet(entries, outputs);
        InputOutputSet[] split = ios.split(ios.getInputs().length / 6 * 5);

        for (double learning = 0.05; learning < 0.2; learning += 0.01) {
            for (int hidden = 6; hidden < 25; hidden++) {
                NeuralNetwork network = new NetworkBuilder()
                        .withLearningRate(learning)
                        .withInputLayer(120)
                        .addHiddenLayer(hidden, ActivationFunction.SIGMOID)
                        .withOutputLayer(1, ActivationFunction.SIGMOID)
                        .build();
                network.trainUntil(split[0].getInputs(), split[0].getOutputs(), 1, Runtime.getRuntime().availableProcessors(), (err, epoch) -> epoch > 1000, 100);


                int errorHuman = 0;
                int errorRobot = 0;
                for (int i = 0; i < split[1].getInputs().length; i++) {
                    double[] input = split[1].getInputs()[i];
                    double pred = network.binaryPredict(input)[0];
                    double sup = split[1].getOutputs()[i][0];

                    if (pred != sup) {
                        if (pred == 0) {
                            errorHuman++;
                        } else {
                            errorRobot++;
                        }
                    }
                }

                int errors = errorHuman + errorRobot;
                double accuracy = (1.0 - errors / (double) split[1].getInputs().length);

                pos.printf("For this neural network we used %d hidden layers with a %.2f learning rate\n", hidden, learning);
                pos.printf("On %d entries we had %d wrong which is a %.2f accuracy rate\n", split[1].getInputs().length, errors,
                        accuracy);
                pos.printf("From these entries we misclassified %d bot paths as human and %d human paths as bot\n\n", errorHuman, errorRobot);
            }
        }
    }

    private static class PointTime {
        private Point point;
        private long time;
    }

}
