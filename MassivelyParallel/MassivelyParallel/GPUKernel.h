#pragma once
#include "Dependencies.h"

class GPUProgram;
class GPUMngr;

class GPUKernel
{
	GPUFRIEND;
private:
	cl_kernel _kernel = nullptr;
	cl_int _currentArgNum = 0;
	cl_uint _dimension = 1;
	const size_t *_global_work_size;
	const size_t *_local_work_size;

public:
	GPUKernel(char *kernelName, GPUProgram &program);
	GPUKernel();
	~GPUKernel();

	void initialize(char *kernelName, GPUProgram &program);
	void resetArgs();
	void addArgBuffer(GPUMem &buffer);

	void execute();

	void setDimension(cl_uint dimension);
	void setGlobalWorkSize(const size_t *global_work_size);
	void setLocalWorkSize(const size_t *local_work_size);
};

