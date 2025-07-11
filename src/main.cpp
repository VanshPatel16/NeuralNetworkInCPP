#include <iostream>
#include "..\include\Eigen\Dense"
#include "..\include\NeuralNetwork.h"
#include "..\include\functions.h"
using namespace Eigen;

int main() {
    std::vector<VectorXd> data = {
        VectorXd{{0,0}},
        VectorXd{{0,1}},
        VectorXd{{1,0}},
        VectorXd{{1,1}}
    };//4 * 2.
    
    std::vector<VectorXd> labels = {
    VectorXd::Constant(1, 0.0),
    VectorXd::Constant(1, 1.0),
    VectorXd::Constant(1, 1.0),
    VectorXd::Constant(1, 0.0)
    };

    // vector is stored as a column vector. -> 
        // std::cout << data << std :: endl;
        // std::cout << labels << std :: endl;
        // std::cout << labels.transpose() * data << std :: endl;
    Layer layer1(2,4,functions::sigmoid,functions::sigmoid_derivative);
    Layer layer2(4,1,functions::sigmoid,functions::sigmoid_derivative);
    NeuralNetwork nn({layer1,layer2});
    nn.train(data,labels,0.01,100);
    
    std::vector<double> y = { 0.0, 1.0, 1.0, 0.0 };
    int correct = 0;
    for (int i = 0; i < 4; ++i) {
    // 1) forward pass
        nn.forward(data[i]);

        // 2) raw network output (sigmoid gives [0,1])
        double raw = nn.get_output()(0);

        // 3) threshold to binary
        int pred = (raw >= 0.5) ? 1 : 0;

        // 4) print & compare
        std::cout << "[" << data[i].transpose() << "] -> "
                    << raw << " -> " << pred
                    << "   (expected " << labels[i] << ")\n";

        if (pred == static_cast<int>(y[i]))
            ++correct;
    }
        
    // 5) overall accuracy
    std::cout << "Accuracy: "
            << (100.0 * correct / 4.0) << "%\n";


}