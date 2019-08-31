#ifndef NEURALNETWORK
#define NEURALNETWORK

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "Neuron.h"

using namespace std;

class NeuralNetwork {
private:
  Neuron zero;
  vector<Neuron> network;
  int numInputs;
  int numLayers;
public:
  NeuralNetwork();
  void addNeuron(string name, vector<double> weights, double inputs);
  double feedForward(vector<double> inputs, string outputName);
  string getNumNeurons();
  void setNumInputs(int i);
  void setNumLayers(int l);
  int getNumInputs();
  int getNumLayers();
  void train(int epoch, int interval, vector<vector<double> >, vector<double> real);
  double deriv_sigmoid(double x);
  double mse_loss(vector<double> r, vector<double> p);
};

double randomize();
vector<double> randomizedVector();


#endif
