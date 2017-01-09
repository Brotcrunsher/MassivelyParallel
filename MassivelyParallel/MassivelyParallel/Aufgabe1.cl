__kernel void calcStatisticAtomic(__global int* in, int length, __global int* out)
{
	int locId = get_local_id(0);
	int grpId = get_group_id(0);
	int gloId = get_global_id(0);

	int startIndex = 8 * locId;
	int endIndex = 8 * (locId + 1);

	local int counts[256];

	for (int i = startIndex; i < endIndex; i++) {
		counts[i] = 0;
		out[i] = 0;
	}
	for (int i = 0; i < 256; i++) {
		if (gloId * 256 + i < length) {
			int Index = gloId * 3 * 256 + i * 3;
			float Y = 0.2126f * in[Index + 0] + 0.7152f * in[Index + 1] + 0.0722f * in[Index + 2];
			atomic_inc(&counts[(int)Y]);
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	for (int i = startIndex; i < endIndex; i++) {
		atomic_add(&out[i], counts[i]);
	}
}

__kernel void calcStatistic(__global int* in, int length, __global int* out)
{
	int locId = get_local_id(0);
	int grpId = get_group_id(0);
	int gloId = get_global_id(0);

	local int counts[32][256];

	for (int i = 0; i < 256; i++) {
		counts[locId][i] = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 0; i < 256; i++) {
		//TODO Jump instead of running through
		if (gloId * 256 + i < length) {
			int Index = gloId * 3 * 256 + i * 3;
			float Y = 0.2126f * in[Index + 0] + 0.7152f * in[Index + 1] + 0.0722f * in[Index + 2];
			Y = clamp(Y, 0.f, 255.f);
			counts[locId][(int)Y] += 1;
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	//TODO Jump instead of running through
	int startIndex = 8 * locId;
	int endIndex = 8 * (locId + 1);
	for (int i = startIndex; i < endIndex; i++) {
		for (int k = 1; k < 32; k++) {
			counts[0][i] += counts[k][i];
		}
		out[i + 256 * grpId] = counts[0][i];
	}
}

__kernel void reduceStatistic(__global int* in, int length)
{
	int gloId = get_global_id(0);

	for (int i = gloId + 256; i < length; i += 256) {
		in[gloId] += in[i];
	}
}