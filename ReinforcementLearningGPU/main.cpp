#include "Network.h"

#include <cmath>
#include <iostream>


int main() {

    int layer_sizes[] = { 1,10,5,5,10,1 };

    Network net(layer_sizes, 6);
    net.train(1024, 50000, 0.01, 0.99, 0.05, true);


    return 0;
}