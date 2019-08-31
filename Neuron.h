#ifndef NEURON_H
#define NEURON_H

#include <iostream>
#include <string>
#include <cmath>
#include <vector>

using namespace std;

class Neuron {
private:
  string name;
  vector<double> weights;
  double bias;
  double sum;
public:
  Neuron(string n, vector<double> w, double b);
  Neuron();
  double feedForward(vector<double> inputs);
  vector<double> getWeights();
  void setWeights(vector<double> w);
  double getBias();
  void setBias(double b);
  string getName();
  void setName(string s);
  double getSum();
  void setSum(double s);
  void toString();
};

double dotProduct(vector<double> weights, vector<double> inputs);
double sigmoid(double x);

#endif
