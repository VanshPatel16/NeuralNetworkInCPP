#ifndef LAYER_H
#define LAYER_H


#include <iostream>
#include "Eigen/Dense"
#include <functional>

using namespace Eigen;


class Layer{
    private:
        MatrixXd weights;
        VectorXd biases;
        VectorXd neuron_values;
        VectorXd neuron_values_activated;
        std :: function<VectorXd(const VectorXd&)>activation_function;
        std :: function<VectorXd(const VectorXd&)>activation_function_derivative;
        VectorXd delta;

    public:

        Layer(int input_size_neurons,int neurons,std::function<VectorXd(const VectorXd&)>activation_function,std::function<VectorXd(const VectorXd&)>activation_function_derivative) : activation_function(activation_function),
            activation_function_derivative(activation_function_derivative){
                assert(input_size_neurons > 0 && neurons > 0);
                weights = MatrixXd :: Random(input_size_neurons,neurons);
                biases = VectorXd :: Random(neurons);
        }

        void forward(const VectorXd&input){
            // Return output
            // input.shape(input_size_neurons,1).
            // weights.shape = (input_size_neurons,neurons).
            // biases.shape = (neurons,1).
            neuron_values = weights.transpose() * input + biases;
            // neuron_values.shape = (neurons,1).
            neuron_values_activated = activation_function(neuron_values);
        }

        VectorXd get_neuron_values_activated(){
            return neuron_values_activated;
        }

        MatrixXd get_weights(){
            return weights;
        }

        void set_delta(const VectorXd&delta){
            this->delta = delta;
        }
        
        VectorXd get_delta(){
            return delta;
        }

        void update_weights(const VectorXd &inputs,double learning_rate){
            weights -= learning_rate * (inputs) * delta.transpose();
            biases -= learning_rate * delta;
        }

        VectorXd derivative_of_activation_function(){
            return activation_function_derivative(neuron_values);
        }
};

#endif