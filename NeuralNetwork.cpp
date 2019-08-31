#include "NeuralNetwork.h"

int main() {

  //Create Data that will be used for training, which is a vector of vectors
  vector<vector<double> > trainingData{ {-3, -2}, {20, 4}, {19, 7}, {-12, -8}, {-1, -2} };
  //Create the real output values that are expected from the Training Data
  vector<double> realValues{ 1, 0, 0, 1, 0};

  //Initialize Network
  NeuralNetwork nn;
  //Print Friendly Comments
  cout << endl;
  cout << endl << "-----Network Initiated -----" << endl;
  //Make Neurons (only 2 hidden and one output created in this example)
  //How many ever neurons are supported for the 1 hidden layer
  //Number of neurons should match number of variable/inputs
  cout << endl << "----- Building Neurons -----" << endl;
  nn.addNeuron("one", randomizedVector(), randomize()); //First Hidden Neuron
  nn.addNeuron("two", randomizedVector(), randomize()); //Second Hidden Neuron
  nn.addNeuron("out", randomizedVector(), randomize()); //Out Neuron
  cout << endl << "----- Network Complete -----" << endl;
  //Start Training
  cout << endl << "----- Training Started -----" << endl;
  //Parameters for .train() are
  //number of training loops, interval to print value,
  //the training Data, and the real output values
  nn.train(300, 10, trainingData, realValues);
  cout << endl << "----- Training Ended -----" << endl;
  //Test the trained neural network with some cases
  vector<double> test1{ -8, -3 };
  vector<double> test2{ 21, 4 };
  cout << endl << "Test1: " << nn.feedForward(test1, "out") << endl;
  cout << endl << "Test2: " << nn.feedForward(test2, "out") << endl;

  cout << endl;
  return 0;
}

//Class Component Definitions

NeuralNetwork::NeuralNetwork() {
  numInputs = 2;
  numLayers = 2;
  cout << endl << "Network has been created with " << numLayers << " layers" << endl;
}

void NeuralNetwork::addNeuron(string name, vector<double> weights, double bias) {
  Neuron n(name, weights, bias);
  network.push_back(n);
}

double NeuralNetwork::feedForward(vector<double> inputs, string outputNeuron) {
  if (inputs.size() == numLayers) {
    int limit = network.size();
    //Can use recursion to support multilayered networks
    vector<double> secondaryInputs;
    for (int i = 0; i < network.size(); i++) {
      if (network.at(i).getName() == outputNeuron) {
        limit = i;
      }
    }
    for (int j = 0; j < limit; j++) {
      secondaryInputs.push_back(network.at(j).feedForward(inputs));
    }
    return network.at(limit).feedForward(secondaryInputs);
  } else {
    cout << endl << "Error: Layer and Input Dimension Mismatch." << endl;
  }
  return 0.0;
}

string NeuralNetwork::getNumNeurons() {
  return network.at(network.size()).getName();
}

void NeuralNetwork::setNumInputs(int i) {
  numInputs = i;
}

void NeuralNetwork::setNumLayers(int l) {
  numLayers = l;
}

int NeuralNetwork::getNumInputs() {
  return numInputs;
}

int NeuralNetwork::getNumLayers() {
  return numLayers;
}

double NeuralNetwork::deriv_sigmoid(double x) {
  return sigmoid(x) * (1 - sigmoid(x));
}

double randomize() {
  srand(time(0));
  return (rand() % 100) / 100.0;
}

vector<double> randomizedVector() {
  vector<double> v{randomize(), randomize()};
  return v;
}

double NeuralNetwork::mse_loss(vector<double> r, vector<double> p) {
  double com = 0;
  if (r.size() == p.size()) {
    for (int i = 0; i < r.size(); i++) {
      com = (r.at(i) - p.at(i)) * (r.at(i) - p.at(i));
      com /= r.size();
    }
    return com;
  } else {
      cout << endl << "MSE Error: Dimension Mismatch" << endl;
  }
  return 0;
}

void NeuralNetwork::train(int epoch, int interval, vector<vector<double> > data, vector<double> real) {
  //Training
  double learnRate = 0.1;
  if (data.size() == real.size()) {
    for (int t = 0; t < epoch; t++) {
      for (int x = 0; x < data.size(); x++) {

          double pred = feedForward(data.at(x), "out");
          double dL_dypred = -2 * (real.at(x) - pred);
          vector<double> weight_derivs;
          vector<double> bias_derivs;
          vector<double> input_derivs;
          for (int k = 0; k < network.size() - 1; k++) {
            input_derivs.push_back(network.at(network.size()-1).getWeights().at(k) * deriv_sigmoid(network.at(network.size()-1).getSum()));
          }
          for (int n = 0; n < network.size() - 1; n++) {
            for (int i = 0; i < data.at(x).size(); i++) {
              weight_derivs.push_back(data.at(x).at(i) * deriv_sigmoid(network.at(n).getSum()));
            }
            bias_derivs.push_back(deriv_sigmoid(network.at(n).getSum()));
          }
          for (int j = 0; j < network.size() - 1; j++) {
            weight_derivs.push_back(network.at(j).getSum() * deriv_sigmoid(network.at(network.size()-1).getSum()));
          }
          bias_derivs.push_back(deriv_sigmoid(network.at(network.size()-1).getSum()));

          for (int e = 0; e < network.size() - 1; e++) {
            vector<double> tempWeights;
            for (int w = 0; w < network.size() - 1; w++) {
              double tempWeight = network.at(e).getWeights().at(w) - ( learnRate * dL_dypred * input_derivs.at(e) * weight_derivs.at(2 * e + w) );
              tempWeights.push_back(tempWeight);
            }
            network.at(e).setWeights(tempWeights);
          }
          for (int b = 0; b < network.size() - 1; b++) {
            double tempBias = network.at(b).getBias() - ( learnRate * dL_dypred * input_derivs.at(b) * bias_derivs.at(b) );
            network.at(b).setBias(tempBias);
          }
          vector<double> outWeights;
          for (int o = 0; o < network.size() - 1; o++) {
            double outWeight = network.at(network.size()-1).getWeights().at(o) - ( learnRate * dL_dypred * weight_derivs.at((network.size()-1) * 2 + o) );
            outWeights.push_back(outWeight);
          }
          network.at(network.size()-1).setWeights(outWeights);
          double outBias = network.at(network.size()-1).getBias() - ( learnRate * dL_dypred * bias_derivs.at(bias_derivs.size()-1));

      }

      if (t %  interval == 0) {
        vector<double> preds;
         for (int f = 0; f < data.size(); f++) {
           preds.push_back(feedForward(data.at(f), "out"));
         }
        double loss = mse_loss(real, preds);
        cout << endl << "Epoch: " << t << " Loss: " << loss << endl;
      }

    }
  } else {
    cout << endl << "Error: Dimension Mismatch during Training" << endl;
  }

}
