#include "stdafx.h"
#include "GPUMem.h"
#include "GPUMngr.h"


GPUMem::GPUMem(cl_mem_flags flags, size_t size, void * host_ptr)
{
	_mem = clCreateBuffer(GPUMngr::instance._context, flags, size, host_ptr, NULL);
	this->_size = size;
}

GPUMem::~GPUMem()
{
	if (_mem != nullptr) {
		cl_int err = clReleaseMemObject(_mem);
		checkErr(err, "Mem Release");
	}
}

void GPUMem::read(void * output)
{
	cl_int err = clEnqueueReadBuffer(GPUMngr::instance._queue, this->_mem, CL_TRUE, 0, _size, output, 0, NULL, NULL);
	checkErr(err, "Reading a buffer failed!");
}
