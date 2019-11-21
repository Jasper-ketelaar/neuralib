package nl.yasper.neuralib;

import java.io.*;
import java.nio.Buffer;

public class PrintAndStoreStream extends PrintStream {

    private final File log;

    public PrintAndStoreStream(File log) {
        super(System.out);
        this.log = log;
    }

    @Override
    public PrintStream printf(String format, Object... args) {
        try {
            BufferedWriter fw = new BufferedWriter(new FileWriter(log, true));
            fw.write(String.format(format, args));
            fw.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return super.printf(format, args);
    }
}
