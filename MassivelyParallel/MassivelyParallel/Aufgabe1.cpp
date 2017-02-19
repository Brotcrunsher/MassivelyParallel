// MassivelyParallel.cpp : Defines the entry point for the console application.
//


#include "Dependencies.h"
#include "GPUMem.h"
#include "GPUProgram.h"
#include "GPUKernel.h"



namespace a_one {
#define MAXIMGLENGTH (1024 * 64)
	std::random_device rnd;
	std::mt19937_64 rng(rnd());
	std::uniform_int_distribution<cl_int> uniformRand(0, 255);
	std::uniform_int_distribution<size_t> uniformRandSize(1024 * 32, MAXIMGLENGTH);

	GPUProgram* program;
	GPUKernel* kernelCalc;
	GPUKernel* kernelReduce;
	GPUKernel* kernelCalcAtomic;

	struct Pixel {
		cl_int R, G, B;
	};

	void initializeKernels() {
		program = new GPUProgram("Aufgabe1.cl");
		kernelCalc = new GPUKernel("calcStatistic", *program);
		kernelReduce = new GPUKernel("reduceStatistic", *program);
		kernelCalcAtomic = new GPUKernel("calcStatisticAtomic", *program);
	}

	void calcHistogramCPU(const Pixel const *img, const size_t length, cl_int histo[]) {
		memset(histo, 0, sizeof(cl_int) * 256);

		for (size_t i = 0; i < length; i++) {
			const Pixel const* pix = &(img[i]);
			cl_float Y = 0.2126f * pix->R + 0.7152f * pix->G + 0.0722f * pix->B;
			histo[(int)Y]++;
		}
	}

	void calcHistogramGPU(const Pixel const *img, const size_t length, cl_int histo[]) {
		cl_int pixelPerWorkItem = 32;

		size_t global_work_size[1] = { (length + 8191) / 8192 * 8192 / 32 };
		size_t local_work_size[1] = { 32 };

		size_t nr_workgroups = global_work_size[0] / local_work_size[0];
		GPUMem inputBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (length) * sizeof(Pixel), (void *)img);
		GPUMem outputBuffer(CL_MEM_READ_WRITE, (256) * sizeof(cl_int) * nr_workgroups, NULL);

		kernelCalc->resetArgs();
		kernelReduce->resetArgs();

		kernelCalc->addArgBuffer(inputBuffer);
		kernelCalc->addArgInt(length);
		kernelCalc->addArgInt(pixelPerWorkItem);
		kernelCalc->addArgBuffer(outputBuffer);


		kernelCalc->setDimension(1);
		kernelCalc->setGlobalWorkSize(global_work_size);
		kernelCalc->setLocalWorkSize(local_work_size);

		cl_event calcEvent = kernelCalc->execute();

		kernelReduce->addArgBuffer(outputBuffer);
		kernelReduce->addArgInt((256) * nr_workgroups);

		kernelReduce->setDimension(1);
		size_t global_work_size_reduce[1] = { 256 };
		size_t local_work_size_reduce[1] = { 32 };
		kernelReduce->setGlobalWorkSize(global_work_size_reduce);
		kernelReduce->setLocalWorkSize(local_work_size_reduce);

		cl_event reduceEvent = kernelReduce->execute(1, &calcEvent);

		//Könnte in ein Array gepackt werden
		clWaitForEvents(1, &calcEvent);
		clWaitForEvents(1, &reduceEvent);

		cl_ulong time_start;
		cl_ulong time_end;

		cl_int status = clGetEventProfilingInfo(calcEvent  , CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &time_start, nullptr);
		checkErr(status, "Event profiling");
		status = clGetEventProfilingInfo(reduceEvent, CL_PROFILING_COMMAND_END  , sizeof(cl_ulong), &time_end  , nullptr);
		checkErr(status, "Event profiling");
		cl_ulong time_taken = time_end - time_start;
		//std::cout << "Time needed: " << time_taken / 1000000.f << std::endl;
		//Average (Pixel per Work Item : Time) :
		//32   : 0.5
		//64   : 0.55
		//128  : 0.65
		//256  : 0.8
		//1024 : 1.2

		outputBuffer.read(histo, 256 * sizeof(cl_int));
	}

	void calcHistogramGPUAtomic(const Pixel const *img, const size_t length, cl_int histo[]) {
		size_t global_work_size[1] = { (length + 8191) / 8192 * 32 };
		size_t local_work_size[1] = { 32 };

		size_t nr_workgroups = global_work_size[0] / local_work_size[0];
		GPUMem inputBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (length) * sizeof(Pixel), (void *)img);
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
			//Could be changed to > 1, however, it can happen that the same bin gets missed twice. >3 should be very, very, very rare.
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

		Pixel* image = new Pixel[MAXIMGLENGTH + 1];
		int maxRuns = 1024;
		for (int i = 0; i < maxRuns; i++) {
			//Move out of loop for performance calculations
			size_t IMGLENGTH = uniformRandSize(rng);
			//IMGLENGTH = MAXIMGLENGTH; //Remove comment for performance calculations
			std::cout << "IMGLENGTH is : " << IMGLENGTH << std::endl;
			cl_int* histoCPU = new cl_int[256];
			cl_int* histoGPU = new cl_int[256];
			for (int i = 0; i < 256; i++) {
				histoCPU[i] = 0;
				histoGPU[i] = 0;
			}
			for (int i = 0; i < MAXIMGLENGTH + 1; i++) {
				image[0].R = image[0].G = image[0].B = 0;
			}
			radomizeImage(image, IMGLENGTH);


			calcHistogramCPU(image, IMGLENGTH, histoCPU); //62.04ms
			//calcHistogramGPU(image, IMGLENGTH, histoGPU); //64.37ms
			calcHistogramGPUAtomic(image, IMGLENGTH, histoGPU); //64.77ms

			checkWinCondition(histoCPU, histoGPU);




			float percentageDone = i / (float)maxRuns * 100.f;
			std::cout << percentageDone << "%\n";
			delete[] histoCPU;
			delete[] histoGPU;
		}
		delete[] image;

		std::cout << "I did it! :) VICTORY!" << std::endl;

		std::chrono::high_resolution_clock::time_point ende = std::chrono::high_resolution_clock::now();
		std::chrono::high_resolution_clock::duration diff = ende - start;
		__int64 ms = std::chrono::duration_cast<std::chrono::milliseconds>(diff).count();
		std::cout << "Time taken: " << ms << "ms" << std::endl;

		while (true);
		return 0;
	}
}


int aufgabe1() {
	return a_one::aufgabe1();
}

