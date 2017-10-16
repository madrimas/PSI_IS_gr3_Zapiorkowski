/**
 * Created by madrimas on 15.10.2017.
 */

import java.util.Random;

public class Perceptron {
    double[] weightVector;
    double thresholdComputation;

    public void perceptron(double[][] inputData, int[] outputData, double thresholdComputation, double learningRate, int numberOfEpochs) {
        this.thresholdComputation = thresholdComputation;
        int lengthOfInputs = inputData[0].length;
        int lengthOfOutputs = outputData.length;
        weightVector = new double[lengthOfInputs]; //creating weightVector of input data size
        Random randomNumbers = new Random(); //creating random number generator

        int epochsCounter = 0;
        int globalError;
        int localError;
        int outcome;

        for (int i = 0; i < lengthOfInputs; i++) {
            weightVector[i] = randomNumbers.nextDouble(); //complete vector with random doubles (0-1)
        }

        do {
            epochsCounter++;
            globalError = 0;
            for (int i = 0; i < lengthOfOutputs; i++) {
                outcome = calculateThreshold(inputData[i]);//threshold computations
                localError = outputData[i] - outcome;//compute local error
                globalError += localError;//adding local error to global error
                for (int j = 0; j < lengthOfInputs; j++) {
                    double delta = learningRate * inputData[i][j] * localError;//compute new weight
                    weightVector[j] += delta;
                }
            }
        } while ((globalError != 0) && (epochsCounter < numberOfEpochs));//
    }

    public int calculateThreshold(double[] inputData) {
        double sum = 0.0;
        for (int i = 0; i < inputData.length; i++) {
            sum += weightVector[i] * inputData[i];
        }
        return sum >= thresholdComputation ? 1 : 0;//validation test
    }
}
