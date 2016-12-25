#include "Dependencies.h"
#include "GPUProgram.h"
#include "GPUKernel.h"
#include "GPUMem.h"


namespace a_two {
	std::random_device rnd;
	std::mt19937_64 rng(rnd());
	std::uniform_int_distribution<cl_int> uniformRand(0, 10);

#define ARRLENGTH 256

	GPUProgram* program;
	GPUKernel* baseAlgo;
	GPUKernel* extendedAlgo;
	GPUKernel* kernelCalcAtomic;

	void initializeKernels() {
		program = new GPUProgram("Aufgabe2.cl");
		baseAlgo = new GPUKernel("baseAlgo", *program);
		extendedAlgo = new GPUKernel("extendedAlgo", *program);
	}

	void execBaseAlgo(const cl_int const * arr, cl_int* prefix) {
		size_t global_work_size[1] = { 128 };
		size_t local_work_size[1] = { 128 };

		GPUMem inputBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (256) * sizeof(cl_int), (void *)arr);
		GPUMem outputBuffer(CL_MEM_READ_WRITE, (256) * sizeof(cl_int), NULL);

		baseAlgo->resetArgs();

		baseAlgo->addArgBuffer(inputBuffer);
		baseAlgo->addArgBuffer(outputBuffer);


		baseAlgo->setDimension(1);
		baseAlgo->setGlobalWorkSize(global_work_size);
		baseAlgo->setLocalWorkSize(local_work_size);

		baseAlgo->execute();

		outputBuffer.read(prefix, 256 * sizeof(cl_int));
	}

	void fillRandom(cl_int* arr, size_t length) {
		for (int i = 0; i < length; i++) {
			arr[i] = uniformRand(rng);
		}
	}

	void prefixCPU(const cl_int const* arr, cl_int* prefixArr, size_t length) {
		if (length == 0) return;

		prefixArr[0] = 0;

		for (int i = 1; i < length; i++) {
			prefixArr[i] = prefixArr[i - 1] + arr[i - 1];
		}
	}

	int aufgabe2() {
		initializeKernels();
		cl_int* arr = new cl_int[ARRLENGTH];
		cl_int* prefixArrCPU = new cl_int[ARRLENGTH];
		cl_int* prefixArrGPU = new cl_int[ARRLENGTH];
		fillRandom(arr, ARRLENGTH);
		prefixCPU(arr, prefixArrCPU, ARRLENGTH);
		execBaseAlgo(arr, prefixArrGPU);

		for (int i = 0; i < ARRLENGTH; i++) {
			std::cout << arr[i] << " ";
		}
		std::cout << std::endl;
		std::cout << std::endl;

		for (int i = 0; i < ARRLENGTH; i++) {
			std::cout << prefixArrCPU[i] << " ";
		}
		std::cout << std::endl;
		std::cout << std::endl;

		for (int i = 0; i < ARRLENGTH; i++) {
			std::cout << prefixArrGPU[i] << " ";
		}
		std::cout << std::endl;
		std::cout << std::endl;


		return 0;
	}
}

int aufgabe2() {
	return a_two::aufgabe2();
}