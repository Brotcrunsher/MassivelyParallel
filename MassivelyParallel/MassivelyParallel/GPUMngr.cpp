#include "GPUMngr.h"
#include "GPUPlatform.h"
#include "GPUProgram.h"

GPUMngr GPUMngr::instance;

GPUMngr::GPUMngr()
{
	initialize();
}

GPUMngr::~GPUMngr()
{
	if (_Platforms != nullptr) {
		delete[] _Platforms;
	}

	if (_queue != nullptr) {
		cl_int err = clReleaseCommandQueue(_queue);
		checkErr(err, "Release Queue");
	}

	if (_context != nullptr) {
		cl_int err = clReleaseContext(_context);
		checkErr(err, "Release Context");
	}
}

void GPUMngr::initialize()
{
	cl_int err = CL_SUCCESS;
	cl_uint numPlatforms = GPUPlatform::getAmountOfPlatforms();

	if (numPlatforms == 0) throw "No Platforms available!";

	{
		_Platforms = new GPUPlatform[numPlatforms];
		cl_platform_id* platformIds = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));
		err = clGetPlatformIDs(numPlatforms, platformIds, NULL);
		for (unsigned int i = 0; i < numPlatforms; i++) {
			_Platforms[i].initialize(platformIds[i]);
		}
		free(platformIds);
	}

	for (unsigned int i = 0; i < numPlatforms; i++) {
		if (_Platforms[i].isCandidate()) {
			_SelectedPlatform = &(_Platforms[i]);
			break;
		}
	}
	if (_SelectedPlatform == nullptr) {
		throw "No correct Platform found!";
	}

	_context = clCreateContext(NULL, 1, _SelectedPlatform->getDeviceIdPtr(), NULL, NULL, NULL);

	_queue = clCreateCommandQueue(_context, _SelectedPlatform->getDeviceId(), 0, NULL);
}
