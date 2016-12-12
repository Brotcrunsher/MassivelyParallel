#pragma once

#include "Dependencies.h"

class GPUPlatform;

class GPUMngr
{
	GPUFRIEND;
private:
	bool _initialized = false;

	GPUPlatform* _Platforms = nullptr;
	GPUPlatform* _SelectedPlatform = nullptr;
	cl_context _context = nullptr;
	cl_command_queue _queue = nullptr;



public:
	GPUMngr();
	~GPUMngr();

	void initialize();
	static GPUMngr instance;
};