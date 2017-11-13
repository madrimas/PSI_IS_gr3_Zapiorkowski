#!/usr/bin/python

from pybrain3.structure import TanhLayer

from pybrain3.structure import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection, RecurrentNetwork
from pybrain3.structure.modules.biasunit import BiasUnit

# network = FeedForwardNetwork()
network = RecurrentNetwork()

inLayer = LinearLayer(35)
# hiddenLayer = SigmoidLayer(5)
hiddenLayer = TanhLayer(5)
outLayer = LinearLayer(1)
bias = BiasUnit()

network.addInputModule(inLayer)
network.addModule(bias)
network.addModule(hiddenLayer)
network.addOutputModule(outLayer)

in_to_hidden = FullConnection(inLayer, hiddenLayer)
bias_to_hidden = FullConnection(bias, hiddenLayer)
hidden_to_out = FullConnection(hiddenLayer, outLayer)

network.addConnection(in_to_hidden)
network.addConnection(bias_to_hidden)
network.addConnection(hidden_to_out)

network.sortModules()