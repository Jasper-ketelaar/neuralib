package nl.yasper.neuralib.network.layer;

import nl.yasper.neuralib.network.perceptron.LearningPerceptron;
import nl.yasper.neuralib.network.perceptron.Perceptron;

import java.security.InvalidParameterException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class PerceptronLayer<T extends Perceptron> implements Iterable<Perceptron> {

    private final List<Perceptron> perceptrons;
    private final int size;

    public PerceptronLayer(int size) {
        this.perceptrons = new ArrayList<>(size);
        this.size = size;
    }

    public void addPerceptron(T perceptron) {
        addPerceptron(perceptron, 1);
    }

    public void addPerceptron(T perceptron, int amount) {
        if (size < amount + perceptrons.size()) {
            throw new InvalidParameterException();
        }

        for (int i = 0; i < amount; i++) {
            perceptrons.add(perceptron.createNew());
        }
    }

    public T getPerceptron(int index) {
        return (T) perceptrons.get(index);
    }

    public double[] predict(double[] input) {
        double[] result = new double[size];
        for (int i = 0; i < size; i++) {
            result[i] = perceptrons.get(i).predict(input);
        }

        return result;
    }

    public int getSize() {
        return size;
    }

    public int getPerceptronSize() {
        return perceptrons.size();
    }

    @Override
    public Iterator<Perceptron> iterator() {
        return perceptrons.iterator();
    }

    @Override
    public PerceptronLayer<T> clone() {
        PerceptronLayer<T> cloned = new PerceptronLayer<>(this.size);
        this.perceptrons.forEach((p) -> {
            Perceptron created = p.createNew();
            if (created instanceof LearningPerceptron) {
                LearningPerceptron createdLearning = (LearningPerceptron) created;
                LearningPerceptron orig = (LearningPerceptron) p;
                for (int i = 0; i < orig.getWeights().length; i++) {
                    createdLearning.setWeight(i, orig.getWeights()[i]);
                }

                createdLearning.setBias(orig.getBias());
            }

            cloned.addPerceptron((T) created);
        });

        return cloned;
    }
}