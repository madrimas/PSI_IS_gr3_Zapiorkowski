#!/usr/bin/python

from pybrain3.supervised.trainers import BackpropTrainer
from pybrain3.tools.validation import Validator

import inputLetters
import network

trainer = BackpropTrainer(network.network, inputLetters.inputLettersDataSet, learningrate=0.1, verbose=True)

trainer.trainEpochs(2000)

testInput = inputLetters.inputLettersDataSet['input']
testTarget = inputLetters.inputLettersDataSet['target']
errorComparator = 0.900

print("Number of training patterns:", len(inputLetters.inputLettersDataSet))

letters = ["A", "B", "C", "D", "I", "F", "G", "H", "K", "U", "a", "b", "c", "d", "f", "h", "m", "o", "w", "z"]

MSE = 0

for i in range(20):
    temp = network.network.activate(testInput[i])
    print("For letter", letters[i], "precision is", temp[0])
    print("MSE:", Validator.MSE(network.network.activate(testInput[i]), testTarget[i]))
    MSE += Validator.MSE(network.network.activate(testInput[i]), testTarget[i])
    if errorComparator < temp:
        print("Letter", letters[i], "is capital letter")
    else:
        print("Letter", letters[i], "is small letter")

print("MSE: ", (MSE/20))