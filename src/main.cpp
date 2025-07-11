#include <iostream>
#include "..\include\Eigen\Dense"

using namespace Eigen;

int main() {
    Matrix<double,4,2> data{
        {0,0},
        {0,1},
        {1,0},
        {1,1}
    };//4 * 2.
    Vector<double,4> labels{
        0,1,1,0
    };// vector is stored as a column vector. -> 
    std :: cout << data << std :: endl;
    std :: cout << labels << std :: endl;
    std :: cout << labels.transpose() * data << std :: endl;
    
}