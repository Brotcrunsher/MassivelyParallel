#define DIR_DOWN 0
#define DIR_UP 1

__kernel void bitonicSort(__global int* inArr, __global int* outArr, int logSize)
{

	int numBatches = logSize;
	int globId = get_global_id(0);

	//TODO make fast ;-)
	if (globId == 0) {
		for (int i = 0; i < 1 << logSize; i++) {
			outArr[i] = inArr[i];
		}
	}

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int batchNr = 0; batchNr < numBatches; batchNr++) {
		int subBatchNr = globId / (1 << batchNr);
		int subBatchSize = 1 << batchNr; //The Amount of workers in one sub batch
		int subBatchIntraId = globId % (subBatchSize); //ID innerhalb eines Sub Batches
		int subBatchPivot = subBatchNr * subBatchSize;
		int dir = DIR_DOWN;
		if (subBatchNr % 2 == 1) {
			dir = DIR_UP;
		}

		for (int subSubBatchNr = 0; subSubBatchNr <= batchNr; subSubBatchNr++) {
			int jumpSize = 1 << (batchNr - subSubBatchNr); //Distanz zwischen hoher und niedriger ID. Entspricht außerdem der größer einer Shift Group
			int shiftGroupNr = subBatchIntraId / jumpSize;
			int shiftGroupIntraId = subBatchIntraId % jumpSize;
			int shiftGroupPivot = shiftGroupNr * jumpSize;
			int highId = subBatchPivot * 2 + shiftGroupPivot * 2 + shiftGroupIntraId;
			int lowId = highId + jumpSize;

			if (dir == DIR_DOWN) {
				if (outArr[highId] > outArr[lowId]) {
					int temp = outArr[highId];
					outArr[highId] = outArr[lowId];
					outArr[lowId] = temp;
				}
			}
			else{
				if (outArr[highId] < outArr[lowId]) {
					int temp = outArr[highId];
					outArr[highId] = outArr[lowId];
					outArr[lowId] = temp;
				}
			}

			barrier(CLK_GLOBAL_MEM_FENCE);
		}
	}
}