from pybrain3 import *

network = FeedForwardNetwork()  # Creating new network

inLayer = LinearLayer(35)  # Creating input layer
hiddenLayer = SigmoidLayer(30)  # Creating hidden layer
outLayer = LinearLayer(20)  # Creating output layer
bias = BiasUnit()  # Initializing Bias

network.addInputModule(inLayer)  # Adding in/out and module layers to the network
network.addModule(bias)
network.addModule(hiddenLayer)
network.addOutputModule(outLayer)

bias_to_hidden = FullConnection(bias, hiddenLayer)  # Creating connection between layers
in_to_hidden = FullConnection(inLayer, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outLayer)

network.addConnection(bias_to_hidden)  # Adding connection to network
network.addConnection(in_to_hidden)
network.addConnection(hidden_to_out)

network.sortModules()  # Sorting modules
