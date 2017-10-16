/**
 * Created by madrimas on 15.10.2017.
 */

public class PerceptronTest {

    static int NUM_EPOCHS = 1000000000;//maximum numbers of epochs
    static double LEARNING_RATE = 0.01;
    static double THRESHOLD = 0.2;

    public static void main(String[] args) {
        Perceptron perceptron = new Perceptron();
        double inputData[][] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};//learning data
        int outputData[] = {0, 0, 0, 1};//learning data

        perceptron.perceptron(inputData, outputData, THRESHOLD, LEARNING_RATE, NUM_EPOCHS);
        System.out.println("Correct: 0, Predicted by AI: " + perceptron.calculateThreshold(new double[]{0, 0}));
        System.out.println("Correct: 0, Predicted by AI: " + perceptron.calculateThreshold(new double[]{1, 0}));
        System.out.println("Correct: 0, Predicted by AI: " + perceptron.calculateThreshold(new double[]{0, 1}));
        System.out.println("Correct: 1, Predicted by AI: " + perceptron.calculateThreshold(new double[]{1, 1}));
    }
}