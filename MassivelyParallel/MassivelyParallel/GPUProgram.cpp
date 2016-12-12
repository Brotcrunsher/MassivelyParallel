#include "stdafx.h"
#include "GPUProgram.h"
#include "GPUMngr.h"
#include "GPUPlatform.h"

GPUProgram::GPUProgram(const char * filename)
{
	initialize(filename);
}

GPUProgram::GPUProgram()
{
}


GPUProgram::~GPUProgram()
{
	if (_program != nullptr) {
		cl_int err = clReleaseProgram(_program);
		checkErr(err, "Release Program");
	}
}

void GPUProgram::initialize(const char *filename)
{
	std::string sourceStr;
	cl_int err = convertToString(filename, sourceStr);
	checkErr(err, "File Loading failed!");
	const char *source = sourceStr.c_str();
	size_t sourceSize[] = { strlen(source) };
	this->_program = clCreateProgramWithSource(GPUMngr::instance._context , 1, &source, sourceSize, NULL);

	err = clBuildProgram(this->_program, 1, GPUMngr::instance._SelectedPlatform->getDeviceIdPtr(), NULL, NULL, NULL);
	if (err) {
		char msg[120000];
		clGetProgramBuildInfo(this->_program, GPUMngr::instance._SelectedPlatform->getDeviceId(), CL_PROGRAM_BUILD_LOG, sizeof(msg), msg, NULL);
		std::cerr << "=== build failed ===\n" << msg << std::endl;
		getc(stdin);
		return;
	}
}
