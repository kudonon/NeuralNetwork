Neural: Neruon.o NeuralNetwork.o
	g++ -o Neural Neuron.o NeuralNetwork.o
Neruon.o: Neuron.cpp
	g++ -std=c++11 -c Neuron.cpp
NeuralNetwork.o: NeuralNetwork.cpp
	g++ -std=c++11 -c NeuralNetwork.cpp
clean: 
	rm Neural Neuron.o NeuralNetwork.o
