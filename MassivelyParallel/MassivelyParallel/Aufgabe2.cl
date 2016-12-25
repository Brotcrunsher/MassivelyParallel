__kernel void baseAlgo(__global int* arr, __global int* prefixArr)
{
	int gloId = get_global_id(0);
	//TODO better load this into a local buffer!
	prefixArr[gloId * 2    ] = arr[gloId * 2    ];
	prefixArr[gloId * 2 + 1] = arr[gloId * 2 + 1];

	barrier(CLK_GLOBAL_MEM_FENCE);
	//Upsweep
	for (int d = 0; d < 7; d++) {
		int lowIndex = pown(2, d) - 1 + gloId * pown(2, d + 1);
		int highIndex = pown(2, d + 1) - 1 + gloId * pown(2, d + 1);
		if (highIndex < 256) {
			prefixArr[highIndex] = prefixArr[lowIndex] + prefixArr[highIndex];
		}
		
		
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//Downsweep
	if (gloId == 0) {
		prefixArr[255] = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int d = 7; d >= 0; d--) {
		int lowIndex = pown(2, d) - 1 + gloId * pown(2, d + 1);
		int highIndex = pown(2, d + 1) - 1 + gloId * pown(2, d + 1);
		if (highIndex < 256) {
			int t = prefixArr[lowIndex];
			prefixArr[lowIndex] = prefixArr[highIndex];
			prefixArr[highIndex] = t + prefixArr[highIndex];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}


__kernel void extendedAlgo(__global int* arr, __global int* prefixArr)
{
}