#pragma once
#include "Dependencies.h"

class GPUProgram;
class GPUMngr;

const bool VERBOSE = false;

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
	void addArgInt(cl_int i);

	cl_event execute(cl_uint num_events_in_wait_list = 0, const cl_event *event_wait_list = nullptr);

	void setDimension(cl_uint dimension);
	void setGlobalWorkSize(const size_t *global_work_size);
	void setLocalWorkSize(const size_t *local_work_size);

	
};

