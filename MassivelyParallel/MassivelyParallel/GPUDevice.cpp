#include "stdafx.h"
#include "GPUDevice.h"


GPUDevice::GPUDevice(cl_device_id id)
{
	initialize(_id);
}

GPUDevice::GPUDevice()
{
}


GPUDevice::~GPUDevice()
{
}

cl_device_id * GPUDevice::getDeviceIdPtr()
{
	return &_id;
}

cl_device_id GPUDevice::getDeviceId()
{
	return _id;
}

void GPUDevice::initialize(cl_device_id id)
{
	this->_id = id;
}
