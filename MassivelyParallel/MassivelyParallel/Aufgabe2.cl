int pow_two(int exp) {
	return 1 << exp;
}

__kernel void baseAlgo(__global int* arr, __global int* prefixArr)
{
	int locId = get_local_id(0);

	prefixArr[locId * 2    ] = arr[locId * 2    ];
	prefixArr[locId * 2 + 1] = arr[locId * 2 + 1];

	barrier(CLK_LOCAL_MEM_FENCE);
	//Upsweep
	for (int d = 0; d < 7; d++) {
		int lowIndex = pow_two(d) - 1 + locId * pow_two(d + 1);
		int highIndex = pow_two(d + 1) - 1 + locId * pow_two(d + 1);
		if (highIndex < 256) {
			prefixArr[highIndex] = prefixArr[lowIndex] + prefixArr[highIndex];
		}
		
		
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//Downsweep
	if (locId == 0) {
		prefixArr[255] = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int d = 7; d >= 0; d--) {
		int lowIndex = pow_two(d) - 1 + locId * pow_two(d + 1);
		int highIndex = pow_two(d + 1) - 1 + locId * pow_two(d + 1);
		if (highIndex < 256) {
			int t = prefixArr[lowIndex];
			prefixArr[lowIndex] = prefixArr[highIndex];
			prefixArr[highIndex] = t + prefixArr[highIndex];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}

int powi(int base, int exp) {
	if (exp == 0) return 1;
	int ret = base;
	for (int i = 1; i < exp; i++) {
		ret *= base;
	}
	return ret;
}




__kernel void extendedAlgo(__global int* A, __global int* B, __global int* C)
{
	int grpId = get_group_id(0);
	int locId = get_local_id(0);
	int prefixOffset = grpId * 256;
	baseAlgo(A + prefixOffset, B + prefixOffset);
	
	barrier(CLK_GLOBAL_MEM_FENCE);
	if (locId == 0) {
		C[grpId] = (A + prefixOffset)[255] + (B + prefixOffset)[255];
	}
	barrier(CLK_GLOBAL_MEM_FENCE);
}

__kernel void finalizeExtended(__global int* B, __global int* D, __global int* E) 
{
	int globId = get_global_id(0);
	int grpId = get_group_id(0);

	E[globId * 2    ] = B[globId * 2    ] + D[grpId];
	E[globId * 2 + 1] = B[globId * 2 + 1] + D[grpId];
}