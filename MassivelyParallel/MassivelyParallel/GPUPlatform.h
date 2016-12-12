#pragma once

#include "Dependencies.h"
#include "GPUDevice.h"

#define INFOLENGTH 128

class GPUPlatform
{
	GPUFRIEND;
private:
	char _name[INFOLENGTH];
	char _vendor[INFOLENGTH];
	char _version[INFOLENGTH];

	cl_platform_id _id = 0;
	cl_bool _candidate = CL_FALSE;
	cl_uint _numDevices = 0;
	GPUDevice *_devices = nullptr;
	GPUDevice *_SelectedDevice = nullptr;

public:
	GPUPlatform(cl_platform_id id);
	GPUPlatform();
	~GPUPlatform();

	void printInfo();
	void initialize(cl_platform_id id);
	cl_bool isCandidate();
	cl_device_id* getDeviceIdPtr();
	cl_device_id getDeviceId();

	static cl_uint getAmountOfPlatforms();
};

