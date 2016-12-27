// MassivelyParallel.cpp : Defines the entry point for the console application.
//

#include "Dependencies.h"
#include "GPUMem.h"
#include "GPUProgram.h"
#include "GPUKernel.h"

int aufgabe1();
int aufgabe2();
int aufgabe3();

int main()
{
	/*while (true) {
		const char* input = "Gdkkn";
		size_t strlength = strlen(input);
		std::cout << "input string:" << std::endl;
		std::cout << input << std::endl;
		char *output = (char*)malloc(strlength + 1);

		GPUMem inputBuffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (strlength + 1) * sizeof(char), (void *)input);
		GPUMem outputBuffer(CL_MEM_WRITE_ONLY, (strlength + 1) * sizeof(char), NULL);

		GPUProgram program("HelloWorld_Kernel.cl");
		GPUKernel kernel("helloworld", program);

		kernel.addArgBuffer(inputBuffer);
		kernel.addArgBuffer(outputBuffer);

		size_t global_work_size[1] = { strlength };
		size_t local_work_size[1] = { strlength };

		kernel.setDimension(1);
		kernel.setGlobalWorkSize(global_work_size);
		kernel.setLocalWorkSize(local_work_size);

		kernel.execute();

		outputBuffer.read(output);
		output[strlength] = '\0';	//Add the terminal character to the end of output.
		std::cout << "\noutput string:" << std::endl;
		std::cout << output << std::endl;
		free(output);

		//TODO make all functions take pointers instead of objects
	}*/
	
	aufgabe1();

	while (true);
    return 0;
}

