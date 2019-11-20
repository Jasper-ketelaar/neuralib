package nl.yasper.neuralib.osrs;

import com.google.gson.*;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.awt.*;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class TestRSMouseData {

    private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

    private List<JsonObject> getPathJsonList() throws IOException {
        Path resources = Paths.get("src", "test", "resources", "nl", "yasper", "neuralib", "recorded-data");
        File directory = resources.toFile();
        if (!directory.exists() || !directory.isDirectory()) {
            Assert.fail("Resource directory doesn't exist");
        }

        List<JsonObject> pathJsonList = new ArrayList<>();
        for (File file : Objects.requireNonNull(directory.listFiles((file, name) -> name.endsWith(".act")))) {
            JsonObject parsed = GSON.fromJson(new FileReader(file), JsonObject.class);
            JsonArray data = parsed.getAsJsonArray("data");
            for (JsonElement entry : data) {
                JsonObject dataEntry = entry.getAsJsonObject();
                if (!dataEntry.get("className").getAsString().equals("MousePathData")) {
                    continue;
                }

                pathJsonList.add(dataEntry);
            }
        }

        return pathJsonList;
    }

    private List<JsonObject> convertRsPacketFormat(List<JsonObject> original) {
        List<JsonObject> pathsRsPacket = new ArrayList<>();
        int counter = 0;
        for (int i = 0; i < original.size(); i++) {
            JsonObject path = original.get(i);
            JsonObject newPathObject = new JsonObject();
            JsonArray entries = new JsonArray();
            long last = 0;
            for (JsonElement pathEntry : path.getAsJsonArray("entries")) {
                JsonObject pathEntryObject = pathEntry.getAsJsonObject();
                long time = pathEntryObject.get("time").getAsLong();
                if (time - last < 50) {
                    continue;
                }

                last = time;
                entries.add(pathEntryObject);
                if (entries.size() >= 40) {
                    newPathObject.add(String.format("path_%d", counter++), entries);
                    pathsRsPacket.add(newPathObject);
                    entries = new JsonArray();
                    newPathObject = new JsonObject();
                }
            }

            newPathObject.add(String.format("path_%d", i), entries);
            pathsRsPacket.add(newPathObject);
        }

        return pathsRsPacket;
    }

    private void flattenAndMap(List<JsonObject> rsPacketList) {
        for (int i = 0; i < rsPacketList.size(); i++) {
            JsonObject path = rsPacketList.get(i);
            String key = path.keySet().iterator().next();
            JsonArray pathEntries = path.get(key).getAsJsonArray();

            long time = 0;
            Point point = null;

            for (JsonElement element : pathEntries) {
                PointTime pt = GSON.fromJson(element, PointTime.class);
                if (point == null) {
                    point = pt.point;
                    time = pt.time;
                }

                double velocity = Math.abs(point.distance(pt.point) / (time - pt.time));
                if (Double.isNaN(velocity)) {
                    velocity = 0;
                }

                element.getAsJsonObject().addProperty("velocity", velocity);
                element.getAsJsonObject().addProperty("x", pt.point.x);
                element.getAsJsonObject().addProperty("y", pt.point.y);
                element.getAsJsonObject().remove("time");
                element.getAsJsonObject().remove("point");

                point = pt.point;
                time = pt.time;
            }

            path.remove(key);
            path.add("path_" + i, pathEntries);
        }
    }

    private void normalizeData(List<JsonObject> flattened) {
        int maxX = Integer.MIN_VALUE;
        int maxY = Integer.MIN_VALUE;
        double maxVel = Double.MIN_VALUE;

        for (int i = 0; i < flattened.size(); i++) {
            JsonArray entries = flattened.get(i).getAsJsonArray("path_" + i);
            for (JsonElement entry : entries) {
                JsonObject entryObject = entry.getAsJsonObject();
                maxX = Math.max(maxX, entryObject.get("x").getAsInt());
                maxY = Math.max(maxY, entryObject.get("y").getAsInt());
                maxVel = Math.max(maxVel, entryObject.get("velocity").getAsDouble());
            }
        }

        for (int i = 0; i < flattened.size(); i++) {
            JsonArray entries = flattened.get(i).getAsJsonArray("path_" + i);
            for (JsonElement entry : entries) {
                JsonObject entryObject = entry.getAsJsonObject();
                entryObject.addProperty("x", (double) entryObject.get("x").getAsInt() / (double) maxX);
                entryObject.addProperty("y", (double) entryObject.get("y").getAsInt() / (double) maxY);
                entryObject.addProperty("velocity", entryObject.get("velocity").getAsDouble() / maxVel);
            }
        }
    }

    private void accumulate() throws IOException {
        System.out.println("Accumulating all .act files");

        List<JsonObject> pathJsonList = getPathJsonList();
        List<JsonObject> pathsRsPacket = convertRsPacketFormat(pathJsonList);
        flattenAndMap(pathsRsPacket);
        normalizeData(pathsRsPacket);

        File paths = new File("./paths.json");
        if (!paths.exists() && !paths.createNewFile()) {
            Assert.fail("Cannot create file");
        }

        FileWriter fw = new FileWriter(paths);
        GSON.toJson(pathsRsPacket, fw);
        fw.close();
    }

    @Test
    public void trainHumanNetwork() throws IOException {
        accumulate();
    }

    private static class PointTime {
        private Point point;
        private long time;
    }

}
