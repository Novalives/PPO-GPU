#pragma once

#ifndef NETWORK_H
#define NETWORK_H

#include <curand_kernel.h>

class Network {
public:
	Network(int*, int);
	~Network();
	int train(const int, const int, double, double, double, bool);
	int net_interface(int*, int, double*, double*);

private:

	const int h_network_size;

	int h_neurons_count, h_weights_count, h_neurons_count_nol1;

	//double* h_neurons;
	//double* h_bias, * h_gamma, * h_beta, *h_running_mean, *h_running_variance;
	//double* h_weights;

	int* h_layer_sizes;


};


#endif
