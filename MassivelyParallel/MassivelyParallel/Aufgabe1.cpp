// MassivelyParallel.cpp : Defines the entry point for the console application.
//

#include "Dependencies.h"
#include "GPUMem.h"
#include "GPUProgram.h"
#include "GPUKernel.h"
#include <random>
#include <chrono>


std::random_device rnd;
std::mt19937_64 rng(rnd());
std::uniform_int_distribution<cl_int> uniformRand(0, 255);
std::uniform_int_distribution<size_t> uniformRandSize(1024, 1024 * 32);

GPUProgram* program;
GPUKernel* kernelCalc;
GPUKernel* kernelReduce;
GPUKernel* kernelCalcAtomic;

struct Pixel{
	cl_int R, G, B;
};

void initializeKernels() {
	program = new GPUProgram("Aufgabe1.cl");
	kernelCalc = new GPUKernel("calcStatistic", *program);
	kernelReduce = new GPUKernel("reduceStatistic", *program);
	kernelCalcAtomic = new GPUKernel("calcStatisticAtomic", *program);
}

void calcHistogramCPU(Pixel *img, cl_int length, cl_int histo[]) {
	memset(histo, 0, sizeof(cl_int)*256);

	for (cl_int i = 0; i < length; i++) {
		Pixel* pix = &(img[i]);
		cl_float Y = 0.2126f * pix->R + 0.7152f * pix->G + 0.0722f * pix->B;
		histo[(int)Y]++;
	}
}

void calcHistogramGPU(Pixel *img, cl_int length, cl_int histo[]) {
	size_t global_work_size[1] = { (length + 8191) / 8192 * 32 };
	size_t local_work_size[1] = { 32 };

	size_t nr_workgroups = global_work_size[0] / local_work_size[0];
	GPUMem inputBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (length)* sizeof(Pixel), (void *)img);
	GPUMem outputBuffer(CL_MEM_READ_WRITE, (256) * sizeof(cl_int) * nr_workgroups, NULL);

	kernelCalc->resetArgs();
	kernelReduce->resetArgs();

	kernelCalc->addArgBuffer(inputBuffer);
	kernelCalc->addArgInt(length);
	kernelCalc->addArgBuffer(outputBuffer);


	kernelCalc->setDimension(1);
	kernelCalc->setGlobalWorkSize(global_work_size);
	kernelCalc->setLocalWorkSize(local_work_size);

	kernelCalc->execute();

	kernelReduce->addArgBuffer(outputBuffer);
	kernelReduce->addArgInt(length);

	kernelReduce->setDimension(1);
	size_t global_work_size_reduce[1] = { 256 };
	size_t local_work_size_reduce[1] = { 32 };
	kernelReduce->setGlobalWorkSize(global_work_size_reduce);
	kernelReduce->setLocalWorkSize(local_work_size_reduce);

	kernelReduce->execute();

	outputBuffer.read(histo, 256 * sizeof(cl_int));
}

void calcHistogramGPUAtomic(Pixel *img, cl_int length, cl_int histo[]) {
	size_t global_work_size[1] = { (length + 8191) / 8192 * 32 };
	size_t local_work_size[1] = { 32 };

	size_t nr_workgroups = global_work_size[0] / local_work_size[0];
	GPUMem inputBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (length)* sizeof(Pixel), (void *)img);
	GPUMem outputBuffer(CL_MEM_READ_WRITE, (256) * sizeof(cl_int) * nr_workgroups, NULL);

	kernelCalcAtomic->resetArgs();

	kernelCalcAtomic->addArgBuffer(inputBuffer);
	kernelCalcAtomic->addArgInt(length);
	kernelCalcAtomic->addArgBuffer(outputBuffer);


	kernelCalcAtomic->setDimension(1);
	kernelCalcAtomic->setGlobalWorkSize(global_work_size);
	kernelCalcAtomic->setLocalWorkSize(local_work_size);

	kernelCalcAtomic->execute();

	outputBuffer.read(histo, 256 * sizeof(cl_int));
}

void checkWinCondition(cl_int *histoCPU, cl_int *histoGPU) {
	bool won = true;
	for (int i = 0; i < 256; i++) {
		//Small differences in Floating Point calculations between CPU and GPU are acceptable.
		//Could be changed to > 1, however, it can happen that the same bin gets misses twice. >3 should be very, very, very rare.
		if (abs(histoCPU[i] - histoGPU[i]) > 3) {

			won = false;
			break;
		}
	}
	if (!won) {
		std::cout << "Uh oh! Something went wrong!" << std::endl;
		std::cout << "Different inizes: ";
		for (int i = 0; i < 256; i++) {
			if (histoCPU[i] != histoGPU[i]) {
				std::cout << "[" << i << "] = {" << histoCPU[i] << ", " << histoGPU[i] << "}" << std::endl;
			}
		}
		while (true);
	}
}

void radomizeImage(Pixel *img, cl_int length) {
	for (cl_int i = 0; i < length; i++) {
		img[i].R = uniformRand(rng);
		img[i].G = uniformRand(rng);
		img[i].B = uniformRand(rng);
	}
}


//#define IMGLENGTH (1024*32)

cl_int abs(cl_int x) {
	if (x < 0) return -x;
	return x;
}

int aufgabe1()
{
	initializeKernels();

	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
	int maxRuns = 1024;
	for (int i = 0; i < maxRuns; i++) {
		//Move out of loop for performance calculations
		size_t IMGLENGTH = uniformRandSize(rng);
		//IMGLENGTH = 1024 * 32; //Remove comment for performance calculations
		Pixel* image = new Pixel[IMGLENGTH];
		cl_int histoCPU[256];
		cl_int histoGPU[256];
		radomizeImage(image, IMGLENGTH);


		calcHistogramCPU(image, IMGLENGTH, histoCPU); //483ms
		calcHistogramGPU(image, IMGLENGTH, histoGPU); //1823ms
		//calcHistogramGPUAtomic(image, IMGLENGTH, histoGPU); //1684ms
		
		checkWinCondition(histoCPU, histoGPU);

		


		float percentageDone = i / (float)maxRuns * 100.f;
		std::cout << percentageDone << "%\n";
		delete[] image;
	}

	std::cout << "I did it! :) VICTORY!" << std::endl;

	std::chrono::high_resolution_clock::time_point ende = std::chrono::high_resolution_clock::now();
	std::chrono::high_resolution_clock::duration diff = ende - start;
	int ms = std::chrono::duration_cast<std::chrono::milliseconds>(diff).count();
	std::cout << "Time taken: " << ms << "ms" << std::endl;

	while (true);
	return 0;
}

