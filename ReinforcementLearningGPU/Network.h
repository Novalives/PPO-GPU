#pragma once

#ifndef NETWORK_H
#define NETWORK_H

#include <chrono>
#include <cstdarg> 
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h> 
#include <cstring>
#include <iostream>

class Network {
public:
	Network(int*, int);
	~Network();
	int train(const int, const int, double, double, double, bool);

	class h_debugger {
	public:
		std::chrono::time_point<std::chrono::high_resolution_clock> full_timer;
		std::chrono::time_point<std::chrono::high_resolution_clock> segment_timer;

		bool debug_flag;
		void setup(const bool& debug_flag) {
			this->debug_flag = debug_flag;
			full_timer = std::chrono::high_resolution_clock::now();
			segment_timer = std::chrono::high_resolution_clock::now();
		}

		void print_timer_start(const char* format, ...) {
			if (debug_flag) {
				printf("~Debug~:");
				va_list argptr;
				va_start(argptr, format);
				vfprintf(stdout, format, argptr);
				va_end(argptr);
				printf("\n");

				segment_timer = std::chrono::high_resolution_clock::now();
			}
		}

		void print_timer_end(const char* format, ...) {
			if (debug_flag) {
				printf("~Debug~:");
				va_list argptr;
				va_start(argptr, format);
				vfprintf(stdout, format, argptr);
				va_end(argptr);

				printf(" in %f seconds\n", std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - segment_timer).count() / 1000000000.0f);
			}
		}

		void print_general(const char* format, ...) {
			if (debug_flag) {
				printf("~Debug~:");
				va_list argptr;
				va_start(argptr, format);
				vfprintf(stdout, format, argptr);
				va_end(argptr);
				printf("\n");
			}
		}

		void print_warning(const char* format, ...) {
			if (debug_flag) {
				printf("\n");
				printf("\t~Warning~");
				va_list argptr;
				va_start(argptr, format);
				vfprintf(stdout, format, argptr);
				va_end(argptr);
				printf("\n\n");
			}
		}

		void print_if_cuda_error(cudaError_t potential_error, const char* format, ...) {
			if (debug_flag && potential_error != cudaSuccess) {
				int index = 0;
				int length = strlen(format);
				for (int i = 0; i < length; i++) {
					if (format[i] == '%' && i + 2 < length) {
						if (format[i + 1] == 'c' && format[i + 2] == 'e') {
							index = i;
							break;
						}
					}
				}
				char* c1 = new char[index];
				char* c2 = new char[length - (index + 3)];
				cudaMemcpy(c1, format, (index) * sizeof(char), cudaMemcpyHostToHost);
				cudaMemcpy(c2, format + index + 3, (length - (index + 3)) * sizeof(char), cudaMemcpyHostToHost);

				printf("\n\t~Error~:");

				va_list argptr;
				va_start(argptr, format);
				vfprintf(stderr, c1, argptr);

				printf("%s", cudaGetErrorString(potential_error));

				vfprintf(stderr, c2, argptr);
				va_end(argptr);

				delete[] c1;
				delete[] c2;

				delete c1;
				delete c2;

				printf("\n\n");

			}
		}

		void end(const char* format, ...) {
			if (debug_flag) {
				va_list argptr;
				va_start(argptr, format);
				vfprintf(stdout, format, argptr);
				va_end(argptr);

				printf(" in %f seconds\n", std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - full_timer).count() / 1000000000.0f);
			}
		}

	};

private:

	const int h_network_size;

	int h_neurons_count, h_weights_count, h_neurons_count_nol1;

	int* h_layer_sizes;

	

};


#endif
