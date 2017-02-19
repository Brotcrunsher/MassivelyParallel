#include "Dependencies.h"
#include "GPUProgram.h"
#include "GPUKernel.h"
#include "GPUMem.h"


namespace a_three {
	std::random_device rnd;
	std::mt19937_64 rng(rnd());
	std::uniform_int_distribution<cl_int> uniformRand(0, 1000000);
	std::uniform_int_distribution<size_t> uniformLogSize(7, 10);

	GPUProgram* program;
	GPUKernel* bitonicSort;

	void initializeKernels() {
		program = new GPUProgram("Aufgabe3.cl");
		bitonicSort = new GPUKernel("bitonicSort", *program);
	}

	void sortGPU(cl_int * arr, cl_int logSize) {
		GPUMem inputBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (1 << logSize) * sizeof(cl_int), (void *)arr);
		GPUMem outputBuffer(CL_MEM_READ_WRITE, (1 << logSize) * sizeof(cl_int), NULL);
		size_t global_work_size[1] = { 1<<(logSize-1) };
		size_t local_work_size[1] = { 1<<(logSize-1) };

		bitonicSort->resetArgs();
		bitonicSort->addArgBuffer(inputBuffer);
		bitonicSort->addArgBuffer(outputBuffer);
		bitonicSort->addArgInt(logSize);

		bitonicSort->setDimension(1);
		bitonicSort->setGlobalWorkSize(global_work_size);
		bitonicSort->setLocalWorkSize(local_work_size);

		bitonicSort->execute();

		outputBuffer.read(arr);
	}



	void fillRandom(cl_int* arr, size_t length) {
		for (int i = 0; i < length; i++) {
			arr[i] = uniformRand(rng);
		}
	}

	void sortCPU(cl_int* arr, cl_int logSize) {
		if (logSize == 0) return;

		cl_int arrSize = 1 << logSize;

		//Einfacher Bubblesort
		bool changesDone = true;
		while (changesDone) {
			changesDone = false;
			for (int i = 1; i < arrSize; i++) {
				if (arr[i - 1] > arr[i]) {
					cl_int temp = arr[i];
					arr[i] = arr[i - 1];
					arr[i - 1] = temp;
					changesDone = true;
				}
			}
		}
	}

	void checkWinCondition(cl_int *CPU, cl_int *GPU, const size_t LENGTH) {
		bool won = true;
		for (int i = 0; i < LENGTH; i++) {
			if (CPU[i] != GPU[i]) {

				won = false;
				break;
			}
		}
		if (!won) {
			std::cout << "Uh oh! Something went wrong!" << std::endl;
			std::cout << "Different inizes: ";
			for (int i = 0; i < LENGTH; i++) {
				if (CPU[i] != GPU[i]) {
					std::cout << "[" << i << "] = {" << CPU[i] << ", " << GPU[i] << "}" << std::endl;
				}
			}
			while (true);
		}
	}

	int aufgabe3() {
		initializeKernels();
		while (true) {
			size_t arrLogSize = uniformLogSize(rng);
			size_t arrSize = 1 << arrLogSize;
			cl_int* arrCPU = new cl_int[arrSize];
			cl_int* arrGPU = new cl_int[arrSize];
			std::cout << "Trying log size/size: " << arrLogSize << "/" << arrSize << std::endl;
			fillRandom(arrCPU, arrSize);
			memcpy(arrGPU, arrCPU, sizeof(cl_int) * arrSize);

			sortCPU(arrCPU, arrLogSize);
			std::cout << "Done CPU" << std::endl;
			sortGPU(arrGPU, arrLogSize);
			std::cout << "Done GPU" << std::endl;

			checkWinCondition(arrCPU, arrGPU, arrSize);

			std::cout << "Victory! :)" << std::endl;
			delete[] arrCPU;
			delete[] arrGPU;
		}


		return 0;
	}
}

int aufgabe3() {
	return a_three::aufgabe3();
}