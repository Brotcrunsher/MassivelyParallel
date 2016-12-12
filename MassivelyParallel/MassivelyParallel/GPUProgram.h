#pragma once
#include "Dependencies.h"

class GPUMngr;

class GPUProgram
{
	GPUFRIEND;
private:
	cl_program _program = nullptr;
public:
	GPUProgram(const char *filename);
	GPUProgram();
	~GPUProgram();

	void initialize(const char *filename);
};

