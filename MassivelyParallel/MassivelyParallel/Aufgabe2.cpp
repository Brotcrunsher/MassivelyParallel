#include "Dependencies.h"
#include "GPUProgram.h"
#include "GPUKernel.h"
#include "GPUMem.h"


namespace a_two {
	std::random_device rnd;
	std::mt19937_64 rng(rnd());
	std::uniform_int_distribution<cl_int> uniformRand(0, 10);
	std::uniform_int_distribution<size_t> uniformSize(1, 5000);

//#define ARRLENGTH 1024

	GPUProgram* program;
	GPUKernel* baseAlgo;
	GPUKernel* extendedAlgo;
	GPUKernel* finalizeExtended;

	void initializeKernels() {
		program = new GPUProgram("Aufgabe2.cl");
		baseAlgo = new GPUKernel("baseAlgo", *program);
		extendedAlgo = new GPUKernel("extendedAlgo", *program);
		finalizeExtended = new GPUKernel("finalizeExtended", *program);
	}

	void _execBaseAlgo(GPUMem &inputBuffer, GPUMem &outputBuffer) {
		size_t global_work_size[1] = { 128 };
		size_t local_work_size[1] = { 128 };
		baseAlgo->resetArgs();

		baseAlgo->addArgBuffer(inputBuffer);
		baseAlgo->addArgBuffer(outputBuffer);


		baseAlgo->setDimension(1);
		baseAlgo->setGlobalWorkSize(global_work_size);
		baseAlgo->setLocalWorkSize(local_work_size);

		baseAlgo->execute();
	}

	void execBaseAlgo(const cl_int const * arr, cl_int* prefix) {
		GPUMem inputBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (256) * sizeof(cl_int), (void *)arr);
		GPUMem outputBuffer(CL_MEM_READ_WRITE, (256) * sizeof(cl_int), NULL);

		_execBaseAlgo(inputBuffer, outputBuffer);

		outputBuffer.read(prefix, 256 * sizeof(cl_int));
	}

	void _execExtendedAlgo(GPUMem &BufferA, GPUMem &BufferB, const size_t ALGOLENGTH, const size_t LENGTH, cl_int* prefix, bool isTopLevel = false)
	{
		size_t global_work_size[1] = { ALGOLENGTH / 2 };
		size_t local_work_size[1] = { 128 };

		const size_t CLENGHT = ((ALGOLENGTH / 256) + 255) / 256 * 256;

		GPUMem BufferC(CL_MEM_READ_WRITE, (CLENGHT)* sizeof(cl_int), NULL);
		GPUMem BufferD(CL_MEM_READ_WRITE, (CLENGHT)* sizeof(cl_int), NULL);

		extendedAlgo->resetArgs();

		extendedAlgo->addArgBuffer(BufferA);
		extendedAlgo->addArgBuffer(BufferB);
		extendedAlgo->addArgBuffer(BufferC);

		extendedAlgo->setDimension(1);
		extendedAlgo->setGlobalWorkSize(global_work_size);
		extendedAlgo->setLocalWorkSize(local_work_size);

		extendedAlgo->execute();


		if (CLENGHT > 256) {
			//TODO make recurive!
			//TOPLEVEL MUST BE FALSE!
			//_execExtendedAlgo();
		}
		else {
			_execBaseAlgo(BufferC, BufferD);
		}

		if (isTopLevel) {
			GPUMem BufferE(CL_MEM_READ_WRITE, ALGOLENGTH * sizeof(cl_int), NULL);
			finalizeExtended->resetArgs();
			finalizeExtended->addArgBuffer(BufferB);
			finalizeExtended->addArgBuffer(BufferD);
			finalizeExtended->addArgBuffer(BufferE);

			finalizeExtended->setDimension(1);
			finalizeExtended->setGlobalWorkSize(global_work_size);
			finalizeExtended->setLocalWorkSize(local_work_size);

			finalizeExtended->execute();

			BufferE.read(prefix, LENGTH * sizeof(cl_int));
		}
	}

	void execExtendedAlgo(const cl_int const * arr, cl_int* prefix, const size_t LENGTH) {
		if (LENGTH == 256) {
			execBaseAlgo(arr, prefix);
			return;
		}

		const size_t ALGOLENGTH = ((LENGTH + 255) / 256) * 256;
		const size_t CDLENGHT = ((ALGOLENGTH / 256) + 255) / 256 * 256;

		GPUMem BufferA(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ALGOLENGTH * sizeof(cl_int), (void *)arr);
		GPUMem BufferB(CL_MEM_READ_WRITE, ALGOLENGTH * sizeof(cl_int), NULL);

		_execExtendedAlgo(BufferA, BufferB, ALGOLENGTH, LENGTH, prefix, true);
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

	int aufgabe2() {
		initializeKernels();
		while (true) {
			size_t ARRLENGTH = uniformSize(rng);
			cl_int* arr = new cl_int[ARRLENGTH];
			cl_int* prefixArrCPU = new cl_int[ARRLENGTH];
			cl_int* prefixArrGPU = new cl_int[ARRLENGTH];
			std::cout << "Trying size: " << ARRLENGTH << std::endl;
			fillRandom(arr, ARRLENGTH);
			prefixCPU(arr, prefixArrCPU, ARRLENGTH);
			execExtendedAlgo(arr, prefixArrGPU, ARRLENGTH);

			checkWinCondition(prefixArrCPU, prefixArrGPU, ARRLENGTH);

			std::cout << "Victory! :)" << std::endl;
			delete[] arr;
			delete[] prefixArrCPU;
			delete[] prefixArrGPU;
		}
		

		return 0;
	}
}

int aufgabe2() {
	return a_two::aufgabe2();
}