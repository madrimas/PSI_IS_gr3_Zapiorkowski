from pybrain3.supervised.trainers import BackpropTrainer
from inputLetters import inputDataSet

import inputLetters
import network

letters = ["A", "B", "C", "D", "I", "F", "G", "H", "K", "U", "M", "E", "L", "O", "P", "R", "T", "W", "X", "Y"]

inp = inputDataSet['input']  # Making shortcut to the input section of DataSet

trainer = BackpropTrainer(network.network, dataset=inputLetters.inputDataSet, learningrate=0.05, verbose=True,
                          momentum=0.1)  # verbose = true so printing errors for each epoch

trainer.trainEpochs(5000)  # Training network for X epochs

print("\n\n")
for i in range(20):  # Final print
    print("Dla litery", letters[i], "output wynosi:")
    temp = network.network.activate(inp[i])
    for k in range(20):
        if temp[k] < 0:
            temp[k] *= (-1)  # Only to improve the analysis of the result
    for j in range(20):
        print(temp[j])
    print("\n")
