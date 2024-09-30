#include "Network.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand.h>
#include <cooperative_groups.h>
#include <cuda.h> 
#include <curand_kernel.h>
#include <algorithm>

#include <ctime>
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <chrono>

struct epoch_data_str {
	double* d_epoch_output;
	double* d_epoch_mean, * d_epoch_variance;
	double* d_epoch_x, * d_epoch_x_norm, * d_epoch_y, * d_epoch_neuron_err;

	curandState* d_epoch_state;

	void allocate_memory(int output_size, int mean_size, int x_size, int state_size) {
		cudaMalloc((void**)&d_epoch_output, output_size);

		cudaMalloc((void**)&d_epoch_mean, mean_size);
		cudaMalloc((void**)&d_epoch_variance, mean_size);

		cudaMalloc((void**)&d_epoch_x, x_size);
		cudaMalloc((void**)&d_epoch_x_norm, x_size);
		cudaMalloc((void**)&d_epoch_y, x_size);
		cudaMalloc((void**)&d_epoch_neuron_err, x_size);

		cudaMalloc((void**)&d_epoch_state, state_size);
	}
	void free_memory() {
		cudaFree(d_epoch_output);
		cudaFree(d_epoch_mean);
		cudaFree(d_epoch_variance);
		cudaFree(d_epoch_x);
		cudaFree(d_epoch_x_norm);
		cudaFree(d_epoch_y);
		cudaFree(d_epoch_neuron_err);
		cudaFree(d_epoch_state);
	}
};

struct gradient_str {
	double* d_bias_gradient, * d_gamma_gradient, * d_beta_gradient;
	double* d_weights_gradient;

	void allocate_memory(int neurons_size, int weights_size) {
		cudaMalloc((void**)&d_bias_gradient, neurons_size);
		cudaMalloc((void**)&d_gamma_gradient, neurons_size);
		cudaMalloc((void**)&d_beta_gradient, neurons_size);
		cudaMalloc((void**)&d_weights_gradient, weights_size);
	}
	void free_memory() {
		cudaFree(d_bias_gradient);
		cudaFree(d_gamma_gradient);
		cudaFree(d_beta_gradient);
		cudaFree(d_weights_gradient);
	}
};

//global array elements
__device__ int d_network_size;

__device__ int d_neurons_count, d_weights_count, d_neurons_count_nol1;

__device__ double* d_neurons;
__device__ double* d_bias, * d_gamma, * d_beta, * d_running_mean, * d_running_variance;
__device__ double* d_weights;

__device__ int* d_layer_sizes;

//pointers that live on host memory but are allocated on device
double* d_neurons_host;
double* d_bias_host, * d_gamma_host, *d_beta_host, *d_running_mean_host, *d_running_variance_host;
double* d_weights_host;

int* d_layer_sizes_host;

//training specific data, allocated as needed
__device__ epoch_data_str* d_epochs_data;
__device__ gradient_str* d_gradients;

__global__ void set_device_array(double* arr, double value) {
	arr[blockDim.x * blockIdx.x + threadIdx.x] = value;
}

Network::Network(int* layer_sizes, const int net_size)
	:
	h_network_size(net_size)
{
	//copy network size to device
	cudaMemcpyToSymbol(d_network_size, &net_size, sizeof(int), 0, cudaMemcpyHostToDevice);

	//copy layer_sizes to device memory
	cudaMalloc((void**)&d_layer_sizes_host, net_size * sizeof(int));
	cudaMemcpy(d_layer_sizes_host, layer_sizes, net_size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_layer_sizes, &d_layer_sizes_host, sizeof(int*), 0, cudaMemcpyHostToDevice);
	h_layer_sizes = layer_sizes;

	//get total number of neurons + weights
	for (int i = 0; i < net_size; i++) {
		h_neurons_count += layer_sizes[i];
		if(i != 0) h_weights_count += layer_sizes[i] * layer_sizes[i - 1];
	}
	h_neurons_count_nol1 = h_neurons_count - layer_sizes[0];
		//copy to device
	cudaMemcpyToSymbol(d_neurons_count, &h_neurons_count, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_weights_count, &h_weights_count, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_neurons_count_nol1, &h_neurons_count_nol1, sizeof(int), 0, cudaMemcpyHostToDevice);

	////allocate host memory
	//h_neurons = (double*)malloc(h_neurons_count * sizeof(double));

	//h_bias = (double*)malloc(h_neurons_count_nol1 * sizeof(double));
	//h_gamma = (double*)malloc(h_neurons_count_nol1 * sizeof(double));
	//std::fill(h_gamma, h_gamma + h_neurons_count_nol1, 1.0f);
	//h_beta = (double*)malloc(h_neurons_count_nol1 * sizeof(double));
	//h_running_mean = (double*)malloc(h_neurons_count_nol1 * sizeof(double));
	//h_running_variance = (double*)malloc(h_neurons_count_nol1 * sizeof(double));

	//h_weights = (double*)malloc(h_weights_count * sizeof(double));
	//std::fill(h_weights, h_weights + h_weights_count, 1.0f);

	//allocate host pointers on device
	cudaMalloc((void**)&d_neurons_host, h_neurons_count * sizeof(double));

	cudaMalloc((void**)&d_bias_host, h_neurons_count_nol1 * sizeof(double));
	cudaMalloc((void**)&d_gamma_host, h_neurons_count_nol1 * sizeof(double));
	set_device_array << <1, h_neurons_count_nol1 >> > (d_gamma_host, 1.0f);
	cudaMalloc((void**)&d_beta_host, h_neurons_count_nol1 * sizeof(double));
	cudaMalloc((void**)&d_running_mean_host, h_neurons_count_nol1 * sizeof(double));
	cudaMalloc((void**)&d_running_variance_host, h_neurons_count_nol1 * sizeof(double));

	cudaMalloc((void**)&d_weights_host, h_weights_count * sizeof(double));
	set_device_array << <1, h_weights_count >> > (d_weights_host, 1.0f);

	cudaDeviceSynchronize();

	//copy pointers to global device memory
	cudaMemcpyToSymbol(d_neurons, &d_neurons_host, sizeof(double*));

	cudaMemcpyToSymbol(d_bias, &d_bias_host, sizeof(double*));
	cudaMemcpyToSymbol(d_gamma, &d_gamma_host, sizeof(double*));
	cudaMemcpyToSymbol(d_beta, &d_beta_host, sizeof(double*));
	cudaMemcpyToSymbol(d_running_mean, &d_running_mean_host, sizeof(double*));
	cudaMemcpyToSymbol(d_running_variance, &d_running_variance_host, sizeof(double*));

	cudaMemcpyToSymbol(d_weights, &d_weights_host, sizeof(double*));

}

__global__ void feedforward_layer(double* x, double* means, double* variances, double* x_norm, double* y, double* output, const int batch_size, const int layer_size, const int layer_prior_size, double* weights, double* bias, double* gamma, double* beta, double* running_mean, double* running_variance, const double momentum, const double leaky_relu) {
	int neuron = blockIdx.x;
	int batch = threadIdx.x;
	//forward prior layer
	double* prior_neurons = output - (layer_prior_size * batch_size + batch);
	for (int pn = 0; pn < layer_prior_size; pn++) {
		x[neuron * batch_size + batch] += prior_neurons[pn * batch_size] * weights[neuron * layer_prior_size + pn];
	}
	x[neuron * batch_size + batch] += bias[neuron];
	//calculate mean
	__shared__ double mean;
	mean = 0;
	atomicAdd(&mean, x[neuron * batch_size + batch] / batch_size);
	__syncthreads();
	//calculate variance
	__shared__ double variance;
	variance = 0;
	atomicAdd(&variance, (x[neuron * batch_size + batch] - mean) * (x[neuron * batch_size + batch] - mean) / batch_size);
	__syncthreads();
	if (batch == 0) {
		means[neuron] = mean;
		running_mean[neuron] = running_mean[neuron] * momentum + mean * (1 - momentum);
		variances[neuron] = variance;
		running_variance[neuron] = running_variance[neuron] * momentum + mean * (1 - momentum);
	}
	//normalize
	x_norm[neuron * batch_size + batch] = (x[neuron * batch_size + batch] - mean) / std::sqrt(variance);
	//gamma + beta
	y[neuron * batch_size + batch] = x_norm[neuron * batch_size + batch] * gamma[neuron] + beta[neuron];
	//activation
	output[neuron * batch_size + batch] = (y[neuron * batch_size + batch] > 0 ? y[neuron * batch_size + batch] : y[neuron * batch_size + batch] * leaky_relu);

}

__global__ void backprop_layer(double* x, double* means, double* variances, double* x_norm, double* y, double* output, const int layer, const int batch_size, const int layer_size, const int layer_prior_size, double* weights, double* bias, double* gamma, double* beta, double* running_mean, double* running_variance, const double leaky_relu, double* layer_errors, double* prior_layer_errors, double* gamma_grad, double* beta_grad, double* weight_grad, double* bias_grad) {
	int neuron = blockIdx.x;
	int batch = threadIdx.x;

	//activation derivative - dy
	double dy = layer_errors[neuron * batch_size + batch] * (y[neuron * batch_size + batch] > 0 ? 1 : leaky_relu);

	//gamma + beta gradient
	atomicAdd(gamma_grad + neuron, dy * x_norm[neuron * batch_size + batch]);
	atomicAdd(&beta_grad[neuron], dy);

	//gamma + beta derivative - dx_norm
	double dx_norm = dy * gamma[neuron];

	//variance derivative - dvar
	__shared__ double x_minus_mean_times_dx_norm_sum;
	x_minus_mean_times_dx_norm_sum = 0;
	atomicAdd(&x_minus_mean_times_dx_norm_sum, dx_norm * (x[neuron * batch_size + batch] - means[neuron]));
	__syncthreads();
	double dvar = x_minus_mean_times_dx_norm_sum * -0.5 * (1.0f / std::sqrt(variances[neuron] + 0.0001) * (1.0f / variances[neuron] + 0.0001));

	//mean derivative - dmean
	__shared__ double dx_norm_sum;
	dx_norm_sum = 0;
	__shared__ double x_minus_mean;
	x_minus_mean = 0;
	atomicAdd(&dx_norm_sum, dx_norm);
	atomicAdd(&x_minus_mean, x[neuron * batch_size + batch] - means[neuron]);
	__syncthreads();
	double dmean = dx_norm_sum * (-1.0f / std::sqrt(variances[neuron] + 0.0001)) + dvar * -2.0f * x_minus_mean * (1.0f / batch_size);

	//x derivative - dx
	double dx = dx_norm * (1.0f / std::sqrt(variances[neuron] + 0.0001)) + dvar * 2 * (x[neuron * batch_size + batch] - means[neuron]) * (1.0f / batch_size) + dmean * (1.0f / batch_size);

	//weights + bias derivative
	for (int pn = 0; pn < layer_prior_size; pn++) {
		atomicAdd(&weight_grad[neuron * layer_prior_size + pn], output[pn * layer_prior_size + batch] * dx);
	}
	atomicAdd(&bias_grad[neuron * layer_size + batch], dx);

	//prior output derivative - layer_errors
	if (layer != 1) {
		for (int pn = 0; pn < layer_prior_size; pn++) {
			*(layer_errors - ((layer_prior_size - pn) * batch_size) + batch) = weights[neuron * layer_prior_size + pn] * dx;
		}
	}

}

__global__ void calculate_output_errors(double* inputs, double* outputs, double* neuron_errors, const int batch_size) {
	int neuron = blockIdx.x;
	int batch = threadIdx.x;

	neuron_errors[neuron * batch_size + batch] = 2 * (exp(inputs[0]) - outputs[neuron * batch_size + batch]);
}

__global__ void setup_kernel(curandState* state) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(1234, id, 0, &state[id]);
}

__global__ void generate_uniform(curandState* state, double* arr) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	arr[id] = curand_uniform(&state[id]);
}

__global__ void train_epoch(const int batch_size, double momentum, double leaky_relu, bool debug_flag) {
	unsigned long long stime = 0;
	if (debug_flag && blockIdx.x == 0 && threadIdx.x == 0) {
		stime = clock();
		printf("Debug~\tsetting up local arrays...\n");
	}

	//get local arrays for convience
	double* x, * output, * neuron_err, * x_norm, * y, * mean, * variance;
	curandState* state;

	output = d_epochs_data->d_epoch_output + (d_neurons_count + d_layer_sizes[0]) * batch_size * (blockDim.x * blockIdx.x + threadIdx.x);

	x = d_epochs_data->d_epoch_x + d_neurons_count * batch_size * (blockDim.x * blockIdx.x + threadIdx.x);
	neuron_err = d_epochs_data->d_epoch_neuron_err + d_neurons_count * batch_size * (blockDim.x * blockIdx.x + threadIdx.x);
	x_norm = d_epochs_data->d_epoch_x + d_neurons_count * batch_size * (blockDim.x * blockIdx.x + threadIdx.x);
	y = d_epochs_data->d_epoch_y + d_neurons_count * batch_size * (blockDim.x * blockIdx.x + threadIdx.x);

	mean = d_epochs_data->d_epoch_mean + d_neurons_count * (blockDim.x * blockIdx.x + threadIdx.x);
	variance = d_epochs_data->d_epoch_variance + d_neurons_count * (blockDim.x * blockIdx.x + threadIdx.x);

	state = d_epochs_data->d_epoch_state + d_layer_sizes[0] * (blockDim.x * blockIdx.x + threadIdx.x);

	if (debug_flag && blockIdx.x == 0 && threadIdx.x == 0) {
		printf("Debug~\tfinished setting up local arrays in %llu clock cycles\n", clock() - stime);
		printf("%f\t%f\t%f\t%f\t%f\t%f\t%f\n", *output, *x, *neuron_err, *x_norm, *y, *mean, *variance);
	}

	//generate random numbers as input
	if (debug_flag && blockIdx.x == 0 && threadIdx.x == 0) {
		stime = clock();
		printf("Debug~\tgenerating input values...\n");
	}

	setup_kernel << <d_layer_sizes[0], batch_size >> > (state);
	generate_uniform << <d_layer_sizes[0], batch_size >> > (state, output);

	if (debug_flag && blockIdx.x == 0 && threadIdx.x == 0) {
		printf("Debug~\tfinished generating input values in %llu clock cycles\n", clock() - stime);
	}

	//feedforward through all layers
	int sum = 0;
	int sum2 = 0;
	for (int layer = 1; layer < d_network_size; layer++) {
		if (debug_flag && blockIdx.x == 0 && threadIdx.x == 0) {
			stime = clock();
			printf("Debug~\tstarting feedforward in layer %d...\n", layer);
		}

		feedforward_layer << <d_layer_sizes[layer], batch_size >> > (x + d_layer_sizes[0] * batch_size + sum * batch_size, 
			mean + sum, 
			variance + sum, 
			x_norm + sum * batch_size, 
			y + sum * batch_size, 
			output + d_layer_sizes[0] * batch_size + sum * batch_size, 
			batch_size, d_layer_sizes[layer], d_layer_sizes[layer - 1], 
			d_weights + sum2, 
			d_bias + sum, 
			d_gamma + sum, 
			d_beta + sum, 
			d_running_mean + sum, 
			d_running_variance + sum, 
			momentum,
			leaky_relu);
		sum += d_layer_sizes[layer];
		sum2 += d_layer_sizes[layer] * d_layer_sizes[layer - 1];

		if (debug_flag && blockIdx.x == 0 && threadIdx.x == 0) {
			printf("Debug~\tfinished feedforward in layer %d in %llu clock cycles\n", layer, clock() - stime);
		}
	}

	if (blockIdx.x == 0 && threadIdx.x == 0) {
		printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	}

	//backprop through all the layers

		//set output errors
	if (debug_flag && blockIdx.x == 0 && threadIdx.x == 0) {
		stime = clock();
		printf("Debug~\tcalculating output errors...\n");
	}

	calculate_output_errors << <d_layer_sizes[d_network_size - 1], batch_size >> > (output, output + sum - (d_layer_sizes[d_network_size - 1] * batch_size), neuron_err + sum - (d_layer_sizes[d_network_size - 1] * batch_size), batch_size);
	
	if (debug_flag && blockIdx.x == 0 && threadIdx.x == 0) {
		printf("Debug~\tfinished calculating output errors in %llu clock cycles\n", clock() - stime);
	}

		//backprop
	int sum3 = 0, sum4 = 0;
	for (int layer = d_network_size - 1; layer > 0; layer--) {
		if (debug_flag && blockIdx.x == 0 && threadIdx.x == 0) {
			stime = clock();
			printf("Debug~\tstarting backpropagation on layer %d...\n", layer);
		}

		sum3 += d_layer_sizes[layer];
		sum4 += d_layer_sizes[layer] * d_layer_sizes[layer - 1];
		backprop_layer << <d_layer_sizes[layer], batch_size >> > (x + (sum + d_layer_sizes[0] - sum3) * batch_size,
			mean + (sum - sum3),
			variance + (sum - sum3),
			x_norm + (sum - sum3) * batch_size,
			y + (sum - sum3) * batch_size,
			output + (sum + d_layer_sizes[0] - sum3) * batch_size,
			layer, batch_size, d_layer_sizes[layer], d_layer_sizes[layer - 1],
			d_weights + (sum2 - sum4),
			d_bias + (sum - sum3),
			d_gamma + (sum - sum3),
			d_beta + (sum - sum3),
			d_running_mean + (sum - sum3),
			d_running_variance + (sum - sum3),
			leaky_relu,
			neuron_err + (sum - sum3) * batch_size,
			neuron_err + (sum - sum3) * batch_size - d_layer_sizes[layer - 1] * batch_size,
			d_gradients->d_gamma_gradient + (sum - sum3),
			d_gradients->d_beta_gradient + (sum - sum3),
			d_gradients->d_weights_gradient + (sum2 - sum4),
			d_gradients->d_bias_gradient + (sum - sum3));

		if (debug_flag && blockIdx.x == 0 && threadIdx.x == 0) {
			printf("Debug~\tfinished backpropagation for layer %d in %llu clock cycles\n", layer, clock() - stime);
		}
	}
	if (blockIdx.x == 0 && threadIdx.x == 0) {
		printf("%s\n", cudaGetErrorString(cudaGetLastError()));
	}

	if (debug_flag && blockIdx.x == 0 && threadIdx.x == 0) {
		printf("Debug~\tdone training\n");
	}

}

__global__ void apply_gradients(int neuron_sum, int weight_sum, int layer, double eta) {
	int neuron = threadIdx.x;

	d_bias[neuron_sum + neuron] += d_gradients->d_bias_gradient[neuron_sum + neuron] * -eta;
	d_gamma[neuron_sum + neuron] += d_gradients->d_gamma_gradient[neuron_sum + neuron] * -eta;
	d_beta[neuron_sum + neuron] += d_gradients->d_beta_gradient[neuron_sum + neuron] * -eta;

	for (int j = 0; j < d_layer_sizes[layer - 1]; j++) {
		d_weights[weight_sum + neuron * d_layer_sizes[layer] + j] += d_gradients->d_weights_gradient[weight_sum + neuron * d_layer_sizes[layer] + j] * -eta;
	}


}

int Network::train(const int batch_size, const int epochs, double eta, double momentum, double leaky_relu, bool debug_flag) {
	auto stime_nano = std::chrono::high_resolution_clock::now();
	if (debug_flag) {
		printf("Debug~\tinitalizing with:\tLayer Sizes: ");
		for (int i = 0; i < h_network_size; i++) printf("%d ", h_layer_sizes[i]);
		printf("\tBatch Size: %d\t\tEpochs: %d\n", batch_size, epochs);
	}

	auto stime_temp = std::chrono::high_resolution_clock::now();
	if (debug_flag) {
		auto stime_temp = std::chrono::high_resolution_clock::now();
		printf("Debug~\tcalculating free space...\n");
	}

	//get available device memory
	size_t free_mem, total_mem;
	cudaMemGetInfo(&free_mem, &total_mem);
		//assume we can use ~50% of free memory
	size_t usable_memory = free_mem * 0.5f;
		//calculate minimum memory needed outside of epochs
	size_t essential_mem = h_neurons_count * sizeof(double) * 3 + h_weights_count * sizeof(double); //gradients
		//calculate memory needed per epoch
	printf("~%d\t%d\t%d\t%d\t%d\t%d~\n", h_neurons_count, batch_size, sizeof(double), sizeof(curandState), h_weights_count, h_neurons_count_nol1);
	size_t needed_mem_per_epoch = 0;
	needed_mem_per_epoch += h_neurons_count * batch_size * sizeof(double) * 1; //outputs
	needed_mem_per_epoch += h_neurons_count_nol1 * sizeof(double) * 2; //mean and variance
	needed_mem_per_epoch += h_neurons_count_nol1 * batch_size * sizeof(double) * 4; //x, x_norm, y, neuron_err
	needed_mem_per_epoch += h_layer_sizes[0] * batch_size * sizeof(curandState); //random number generator state
	//get max amount of epochs
	size_t max_epochs_sync = ((usable_memory - essential_mem) / needed_mem_per_epoch);
	if (debug_flag) {
		std::cout << "Debug~\tfinished calculating free space in " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - stime_temp).count() / 1000000000.0f << " seconds\n";
		printf("\t\tTotal Space (bytes): %zu | Total Usable Space (bytes): %zu\n", total_mem, free_mem);
		printf("\t\tMax memory able to be utilized by program (bytes): %zu\n", usable_memory);
		printf("\t\tEssential Memory Needed (bytes): %zu | Memory Per Epoch Needed (bytes): %zu\n", essential_mem, needed_mem_per_epoch);
		printf("\t\tMax Number of Epochs running synchronously: %lu\n", max_epochs_sync);
	}

	//get largest layer size
	int h_largest_layer = h_layer_sizes[1];
	for (int i = 2; i < h_network_size; i++) if (h_layer_sizes[i] > h_largest_layer) h_largest_layer = h_layer_sizes[i];
	//make sure we can run enough threads inside the training kernel
	cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, h_largest_layer * 1024 * max_epochs_sync);

	//create stucture for gradients/value storage
	if (debug_flag) {
		stime_temp = std::chrono::high_resolution_clock::now();
		printf("Debug~\tallocating necessary epoch memory on device...\n");
	}

	epoch_data_str* d_epochs_data_host;
	epoch_data_str* h_epochs_data;
	gradient_str* d_gradients_host;
	gradient_str* h_gradients;

	cudaMalloc((void**)&d_epochs_data_host, sizeof(epoch_data_str));
	cudaMalloc((void**)&d_gradients_host, sizeof(gradient_str));
	h_epochs_data = (epoch_data_str*)malloc(sizeof(epoch_data_str));
	h_gradients = (gradient_str*)malloc(sizeof(gradient_str));

	h_gradients->allocate_memory(h_neurons_count_nol1 * sizeof(double), h_weights_count * sizeof(double));
	cudaMemcpy(d_gradients_host, h_gradients, sizeof(gradient_str), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_gradients, &d_gradients_host, sizeof(gradient_str*));

	//calculate how many cycles we will need to run
	size_t epoch_runs = epochs / max_epochs_sync;
	size_t epoch_runs_debug = epoch_runs;
	if (epoch_runs == 0) printf("\n\tWarning: Epoch value of %d is less then the amount of epochs that can run synchonously (%d)\n\n", epochs, max_epochs_sync);
	if (epoch_runs != 0 && epoch_runs * max_epochs_sync != epochs) printf("\n\tWarning: Epoch value of %d is not a multiple of the amount of epochs that can run synchonously (%d)\n\n", epochs, max_epochs_sync);

	//run epochs
	if (debug_flag) {
		stime_temp = std::chrono::high_resolution_clock::now();
		printf("Debug~\tstarting training...\n");
	}

	bool debug_flag_internal = true;
	int run = 1;

	size_t block_epochs = max_epochs_sync / 1024;
	if (epoch_runs != 0 && block_epochs * 1024 != max_epochs_sync) printf("\n\tWarning: Number of epochs running synchonously is not divisible by 1024\n\n");

	if (epoch_runs != 0) {
		printf("~%d\t%d\t%d\t%d\n", h_neurons_count * batch_size * block_epochs * 1024 * sizeof(double),
			h_neurons_count_nol1 * block_epochs * 1024 * sizeof(double),
			h_neurons_count_nol1 * batch_size * block_epochs * 1024 * sizeof(double),
			h_layer_sizes[0] * batch_size * block_epochs * 1024 * sizeof(curandState));
		//allocate device memory
		if (debug_flag) {
			stime_temp = std::chrono::high_resolution_clock::now();
			printf("Debug~\tallocating epoch specific device memory for epoch number of %d...\n", block_epochs * 1024);
		}

		h_epochs_data->allocate_memory(h_neurons_count * batch_size * block_epochs * 1024 * sizeof(double),
									   h_neurons_count_nol1 * block_epochs * 1024 * sizeof(double),
									   h_neurons_count_nol1 * batch_size * block_epochs * 1024 * sizeof(double),
									   h_layer_sizes[0] * batch_size * block_epochs * 1024 * sizeof(curandState));
		cudaMemcpy(d_epochs_data_host, h_epochs_data, sizeof(epoch_data_str), cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(d_epochs_data, &d_epochs_data_host, sizeof(epoch_data_str*));

		if (debug_flag) {
			std::cout << "Debug~\tfinished allocating epoch specific memory in " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - stime_temp).count() / 1000000000.0f << " seconds\n";
		}
	}

	int thread_dim = 1024;
	while (epoch_runs > 0) {
		start_train:
		//train
		train_epoch << <block_epochs, thread_dim >> > (
			batch_size,
			momentum,
			leaky_relu,
			debug_flag_internal);
		cudaError_t status = cudaDeviceSynchronize();
		std::cout << cudaGetErrorString(status) << std::endl;
		//sometimes cuda doesnt like running a lot of threads at once, esspecially with a low batch size, so we can reduce the threads per block to attempt to fix this
		cudaError_t initalization_status = cudaGetLastError();
		std::cout << cudaGetErrorString(initalization_status) << std::endl;
		if (initalization_status == CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES) {
			thread_dim /= 2;
			block_epochs *= 2;
			//reallocate memory just in case
			h_epochs_data->free_memory();
			cudaFree(d_epochs_data_host);
			h_epochs_data->allocate_memory(h_neurons_count* batch_size* block_epochs * 1024 * sizeof(double),
				h_neurons_count_nol1* block_epochs * 1024 * sizeof(double),
				h_neurons_count_nol1* batch_size* block_epochs * 1024 * sizeof(double),
				h_layer_sizes[0] * batch_size* block_epochs * 1024 * sizeof(curandState));
			cudaMemcpy(d_epochs_data_host, h_epochs_data, sizeof(epoch_data_str), cudaMemcpyHostToDevice);
			cudaMemcpyToSymbol(d_epochs_data, &d_epochs_data_host, sizeof(epoch_data_str*));
			printf("\n\tWarning: Cuda initalization out of resources. Reducing threads to %d and increasing block to %d\n\n", thread_dim, block_epochs);
			goto start_train;
		}
		else if (initalization_status != cudaSuccess) {
			printf("\n\tError: Training batch %d failed to initalize with error code: %s\n\n", run, cudaGetErrorString(initalization_status));
			h_epochs_data->free_memory();
			h_gradients->free_memory();
			free(h_epochs_data);
			free(h_gradients);
			cudaFree(d_gradients_host);
			cudaFree(d_epochs_data_host);
			return -1;
		}
		debug_flag_internal = false; //only run the debug code for the first epoch
		epoch_runs -= 1;
		if (status != cudaSuccess) {
			printf("\n\tError: Training batch %d returned error code: %s\n\n", run, cudaGetErrorString(status));
			h_epochs_data->free_memory();
			h_gradients->free_memory();
			free(h_gradients);
			free(h_epochs_data);
			cudaFree(d_gradients_host);
			cudaFree(d_epochs_data_host);
			return -1;
		}
		if (debug_flag) {
			printf("Debug~\tFinishing calculating gradients for %d epochs\n", block_epochs * thread_dim * run);
		}
		run++;
		if (epoch_runs == 0) {
			h_epochs_data->free_memory();
			cudaFree(d_epochs_data_host);
		}
	}

	//calculate how many epochs couldn't be run in max batches
	int remaining_epochs = epochs - (block_epochs * 1024 * epoch_runs_debug);
	printf("~%d\t%d\t%d~\n", epochs, block_epochs, epoch_runs_debug);
	cudaError_t status;
	int remaining_epochs_batch = remaining_epochs / 1024;

	printf("\n%d\n", remaining_epochs_batch * 1024);

	if (remaining_epochs > 1024) { //epochs we can run in thread of 1024
		//allocate memory
		h_epochs_data->allocate_memory(h_neurons_count* batch_size* remaining_epochs_batch * 1024 * sizeof(double),
			h_neurons_count_nol1* remaining_epochs_batch * 1024 * sizeof(double),
			h_neurons_count_nol1* batch_size* remaining_epochs_batch * 1024 * sizeof(double),
			h_layer_sizes[0] * batch_size* remaining_epochs_batch * 1024 * sizeof(curandState));
		cudaMemcpy(d_epochs_data_host, h_epochs_data, sizeof(epoch_data_str), cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(d_epochs_data, &d_epochs_data_host, sizeof(epoch_data_str*));
		//train
		train_epoch << <remaining_epochs_batch, 1024 >> > (
			batch_size,
			momentum,
			leaky_relu,
			debug_flag_internal);
		//free epoch memory
		h_epochs_data->free_memory();
		cudaFree(d_epochs_data_host);
		if (debug_flag) {
			printf("Debug~\tFinishing calculating gradients for %d epochs\n", block_epochs * 1024 * (run - 1) + remaining_epochs_batch * 1024);
		}
		status = cudaDeviceSynchronize();
		if (status != cudaSuccess) {
			printf("\n\tError: Training batch %d returned error code: %s\n\n", run, cudaGetErrorString(status));
			h_gradients->free_memory();
			free(h_gradients);
			free(h_epochs_data);
			cudaFree(d_gradients_host);
			return -1;
		}
	}
	remaining_epochs -= (remaining_epochs / 1024) * 1024; //total number of epochs left should be below 1024
	printf("\n%d\n", remaining_epochs);
	if (remaining_epochs != 0) {
		//allocate memory
		h_epochs_data->allocate_memory(h_neurons_count* batch_size* remaining_epochs * sizeof(double),
			h_neurons_count_nol1* remaining_epochs * sizeof(double),
			h_neurons_count_nol1* batch_size* remaining_epochs * sizeof(double),
			h_layer_sizes[0] * batch_size* remaining_epochs * sizeof(curandState));
		status = cudaMemcpy(d_epochs_data_host, h_epochs_data, sizeof(epoch_data_str), cudaMemcpyHostToDevice);
		std::cout << cudaGetErrorString(status) << std::endl;
		status = cudaMemcpyToSymbol(d_epochs_data, &d_epochs_data_host, sizeof(epoch_data_str*));
		std::cout << cudaGetErrorString(status) << std::endl;
		std::cout << cudaGetErrorString(cudaDeviceSynchronize()) << std::endl;
		//train
		train_epoch << <1, remaining_epochs >> > (
			batch_size,
			momentum,
			leaky_relu,
			debug_flag_internal);
		//free memory
		status = cudaDeviceSynchronize();
		h_epochs_data->free_memory();
		cudaFree(d_epochs_data_host);
		if (status != cudaSuccess) {
			printf("\n\tError: Training batch %d returned error code: %s\n\n", run, cudaGetErrorString(status));
			h_gradients->free_memory();
			free(h_gradients);
			free(h_epochs_data);
			cudaFree(d_gradients_host);
			return -1;
		}
	}

	if (debug_flag) {
		printf("Debug~\tFinishing calculating gradients for %d epochs\n", block_epochs * 1024 * (run - 1) + remaining_epochs_batch * 1024 + remaining_epochs);
	}
	if (debug_flag) {
		std::cout << "Debug~\tfinished calculating gradients with no errors in " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - stime_temp).count() / 1000000000.0f << " seconds\n";
	}

	if (debug_flag) {
		stime_temp = std::chrono::high_resolution_clock::now();
		printf("Debug~\tapplying gradients...\n");
	}
	int n_sum = 0, w_sum = 0;
	for (int l = 1; l < h_network_size; l++) {
		apply_gradients << <1, h_layer_sizes[l] >> > (n_sum, w_sum, l, eta);
		n_sum += h_layer_sizes[l];
		w_sum += h_layer_sizes[l];
	}
	h_gradients->free_memory();
	free(h_gradients);
	free(h_epochs_data);
	cudaFree(d_gradients_host);
	if (debug_flag) {
		std::cout << "Debug~\tfinished applying gradients with error code: " << cudaGetErrorString(cudaDeviceSynchronize()) << " in " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - stime_temp).count() / 1000000000.0f << " seconds\n";
	}

	std::cout << "\n\t Total program time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - stime_nano).count() / 1000000000.0f << " seconds\n\n";
	return 0;
}

Network::~Network() {
	cudaFree(d_neurons_host);
	cudaFree(d_bias_host);
	cudaFree(d_gamma_host);
	cudaFree(d_beta_host);
	cudaFree(d_running_mean_host);
	cudaFree(d_running_variance_host);
	cudaFree(d_weights_host);
	cudaFree(d_layer_sizes_host);
}



