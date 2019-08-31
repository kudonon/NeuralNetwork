#include "Neuron.h"

Neuron::Neuron() {
  name = "NULL";
  for (int i = 0; i < 2; i++) {
    weights.push_back(0);
  }
  bias = 0;
  cout << endl << "Neuron has been created with a name of " << name << endl;
}

Neuron::Neuron(string n, vector<double> w, double b) {
  name = n;
  for (int i = 0; i < w.size(); i++) {
    weights.push_back(w.at(i));
  }
  bias = b;
  cout << endl << "Neuron has been created with a name of " << name << endl;
}

double Neuron::feedForward(vector<double> inputs) {
 sum = dotProduct(weights, inputs) + bias;
 return sigmoid(sum);
}

vector<double> Neuron::getWeights() {
  return weights;
}

void Neuron::setWeights(vector<double> w) {
  weights = w;
}

double Neuron::getBias() {
  return bias;
}

void Neuron::setBias(double b) {
  bias = b;
}

string Neuron::getName() {
  return name;
}

void Neuron::setName(string s) {
  name = s;
}

double Neuron::getSum() {
  return sum;
}

void Neuron::setSum(double s) {
  sum = s;
}

void Neuron::toString() {
  cout << endl << "Weights: ";
  for (int i = 0; i < weights.size(); i++) {
    cout << weights.at(i) << " " << endl;
  }
  cout << "Bias: " << bias << endl;
}

double dotProduct(vector<double> u, vector<double> v) {
  if (u.size() == v.size()) {
    double prod = 0.0;
    for (int i = 0; i < u.size(); i++) {
      prod += u.at(i) * v.at(i);
    }
    return prod;
  } else {
    cout << endl << "Error Computing Dot Product (Dimension Mismatch)" << endl;
  }
  return 0.0;
}

double sigmoid(double x) {
  return (1)/(1 + exp(x));
}
