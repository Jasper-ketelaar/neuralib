package nl.yasper.neuralib.math;

import java.security.InvalidParameterException;

public class ArrayMath {

    public static double dotProduct(double[] vector1, double[] vector2) {
        if (vector1.length != vector2.length) {
            throw new InvalidParameterException(String.format("Vector 1 length %d != %d vector 2 length", vector1.length, vector2.length));
        }

        double sum = 0;
        for (int i = 0; i < vector1.length; i++) {
            sum += vector1[i] * vector2[i];
        }

        return sum;
    }
}
