#include "stdafx.h"
#include "GPUKernel.h"
#include "GPUProgram.h"
#include "GPUMem.h"
#include "GPUMngr.h"

GPUKernel::GPUKernel()
{
}


GPUKernel::GPUKernel(char * kernelName, GPUProgram & program)
{
	initialize(kernelName, program);
}

GPUKernel::~GPUKernel()
{
	if (this->_kernel != nullptr) {
		cl_int err = clReleaseKernel(_kernel);
		checkErr(err, "Release Kernel");
	}
}

void GPUKernel::initialize(char * kernelName, GPUProgram & program)
{
	_kernel = clCreateKernel(program._program, kernelName, NULL);
}

void GPUKernel::resetArgs()
{
	_currentArgNum = 0;
}

void GPUKernel::addArgBuffer(GPUMem & buffer)
{
	if(VERBOSE)
		std::cout << "Adding Buffer " << buffer._mem << " at location " << _currentArgNum << std::endl;
	cl_int err = clSetKernelArg(_kernel, _currentArgNum, sizeof(cl_mem), (void*)&(buffer._mem));
	checkErr(err, "Load Buffer");
	_currentArgNum++;
}

void GPUKernel::addArgInt(cl_int i)
{
	if(VERBOSE)
		std::cout << "Adding int " << i << " at location " << _currentArgNum << std::endl;
	cl_int err = clSetKernelArg(_kernel, _currentArgNum, sizeof(cl_int), &i);
	checkErr(err, "Load Buffer");
	_currentArgNum++;
}

void GPUKernel::execute()
{
	cl_int status= clEnqueueNDRangeKernel(GPUMngr::instance._queue, this->_kernel, this->_dimension, NULL, this->_global_work_size, this->_local_work_size, 0, NULL, NULL);
}

void GPUKernel::setDimension(cl_uint dimension)
{
	this->_dimension = dimension;
}

void GPUKernel::setGlobalWorkSize(const size_t * global_work_size)
{
	this->_global_work_size = global_work_size;
}

void GPUKernel::setLocalWorkSize(const size_t * local_work_size)
{
	this->_local_work_size = local_work_size;
}
