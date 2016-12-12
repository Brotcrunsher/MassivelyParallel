#include "stdafx.h"
#include "GPUPlatform.h"


GPUPlatform::GPUPlatform(cl_platform_id id)
{
	initialize(id);
}

GPUPlatform::GPUPlatform()
{
}


GPUPlatform::~GPUPlatform()
{
	if (_devices != nullptr) {
		delete[] _devices;
	}
}

void GPUPlatform::printInfo()
{
	std::cout 
		<< "PlatformInfo[" << std::endl 
			<< "\tName:    " << _name << std::endl 
			<< "\tVendor:  " << _vendor << std::endl 
			<< "\tVersion: " << _version << std::endl 
		<< "]" << std::endl;
}

void GPUPlatform::initialize(cl_platform_id id)
{
	this->_id = id;
	cl_int err = CL_SUCCESS;
	err = clGetPlatformInfo(_id, CL_PLATFORM_VENDOR, INFOLENGTH, _vendor, NULL);
	checkErr(err, "Couldn't read vendor");
	err = clGetPlatformInfo(_id, CL_PLATFORM_NAME, INFOLENGTH, _name, NULL);
	checkErr(err, "Couldn't read name");
	err = clGetPlatformInfo(_id, CL_PLATFORM_VERSION, INFOLENGTH, _version, NULL);
	checkErr(err, "Couldn't read version");

	_numDevices = 0;
	err = clGetDeviceIDs(_id, CL_DEVICE_TYPE_GPU, 0, NULL, &_numDevices);
	if (err == CL_DEVICE_NOT_FOUND) {
		//The platform is propably a cpu platform. So this platform is no candidate for gpu calculations.
		_candidate = CL_FALSE;
		return;
	}
	checkErr(err, "Something went wrong with device creation!"); //Other error checking

	if (_numDevices == 0) {
		//Should never happen but if there ever is a Platform without any devices, than that platform obviously is not a candidate.
		_candidate = CL_FALSE;
		return;
	}

	cl_device_id *devices = (cl_device_id*)malloc(_numDevices * sizeof(cl_device_id));
	err = clGetDeviceIDs(_id, CL_DEVICE_TYPE_GPU, _numDevices, devices, NULL);
	_devices = new GPUDevice[_numDevices];
	for (unsigned int i = 0; i < _numDevices; i++) {
		_devices[i].initialize(devices[i]);
	}
	free(devices);

	//Just pick the first one for now
	//TODO Choose the "best device" - what ever that means.
	_SelectedDevice = &(_devices[0]);

	_candidate = CL_TRUE;
}

cl_bool GPUPlatform::isCandidate()
{
	return _candidate;
}

cl_device_id * GPUPlatform::getDeviceIdPtr()
{
	return _SelectedDevice->getDeviceIdPtr();
}

cl_device_id GPUPlatform::getDeviceId()
{
	return _SelectedDevice->getDeviceId();
}

cl_uint GPUPlatform::getAmountOfPlatforms()
{
	cl_uint numPlatforms;
	cl_int err = clGetPlatformIDs(0, NULL, &numPlatforms);
	checkErr(err, "Couldn't read Amount of Platforms!");
	return numPlatforms;
}
