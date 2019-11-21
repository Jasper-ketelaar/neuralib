package nl.yasper.neuralib.osrs.struct.flat;

public class FlattenedMousePath {

    private FlattenedMouseEntry[] entries;
    private boolean human;

    public FlattenedMousePath(FlattenedMouseEntry[] entries, boolean human) {
        this.entries = entries;
        this.human = human;
    }

    public boolean isHuman() {
        return human;
    }

    public FlattenedMouseEntry[] getEntries() {
        return entries;
    }
}
