#pragma once

#include "Dependencies.h"

#define checkErr(error, msg) if(error != CL_SUCCESS){\
		std::cout << msg << std::endl; \
		std::cout << "Error Code: " << error << std::endl; \
		__debugbreak();\
	}

/* convert the kernel file into a string */
int convertToString(const char *filename, std::string& s);

#define GPUFRIEND \
	friend class GPUDevice;\
	friend class GPUMngr;\
	friend class GPUPlatform;\
	friend class GPUProgram;\
	friend class GPUKernel;\
	friend class GPUMem;