
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>
#include <cmath>
#include <iostream>

#include "Network.h"


int main(){

    int layer_sizes[] = {1,10,5,5,10,1};

    Network net(layer_sizes, 6);
    net.train(1024, 50000, 0.01, 0.99, 0.05, true);
    
    //for (int i = 0; i < 100; i++) {
    //    double input[1] = { (double)rand() / RAND_MAX };
    //    double output[1];
    //    net.net_interface(layer_sizes, 6, input, output);
    //    std::cout << exp(input[0]) << "\t" << output[0] << std::endl;
    //}
   

    return 0;
}
