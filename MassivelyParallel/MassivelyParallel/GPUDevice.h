#pragma once
#include "Dependencies.h"

class GPUDevice
{
	GPUFRIEND;
private:
	cl_device_id _id = nullptr;
public:
	GPUDevice(cl_device_id id);
	GPUDevice();
	~GPUDevice();

	cl_device_id* getDeviceIdPtr();
	cl_device_id getDeviceId();

	void initialize(cl_device_id id);
};

