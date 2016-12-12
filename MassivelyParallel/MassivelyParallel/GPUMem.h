#pragma once
#include "Dependencies.h"
#include "Utils.h"

class GPUMem
{
	GPUFRIEND;
private:
	cl_mem _mem = nullptr;
	size_t _size = 0;

public:
	GPUMem(cl_mem_flags flags, size_t size, void* host_ptr);
	~GPUMem();

	void read(void* output);
};

