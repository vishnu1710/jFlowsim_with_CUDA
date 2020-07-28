#include "LB.h"
extern "C"
__global__ void LBMMovingWallBC(int bc_type, float *df, int LX, int LY, float vx, float vy, float vScale)
{

//////////////////////////////////////////////////////////////////////////////////////////////////
	
	int tIdx = threadIdx.x;
	int tIdy = threadIdx.y;
	
	int tx = tIdx + blockIdx.x * blockDim.x;
	int ty = tIdy + blockIdx.y * blockDim.y;

	float F_IN_E, F_IN_N, F_IN_W, F_IN_S, F_IN_NE, F_IN_NW, F_IN_SW, F_IN_SE;
	int nodeIndex;
//////////////////////////////////////////////////////////////////////////////////////////////////
if(tx < LX && ty < LY)
{
	
	
	if(bc_type == EAST){
	
	nodeIndex = ((LX -1) + ty * LX) * 9;
		
		F_IN_E = df[nodeIndex + 1] - 2/3 * vx/vScale;
		F_IN_SE = df[nodeIndex + 8] - 1. / 6. * (vx - vy)/vScale;
		F_IN_NE = df[nodeIndex + 5] - 1. / 6. * (vx + vy)/vScale;
		
		df[nodeIndex + 2] = F_IN_E;
		df[nodeIndex + 7] = F_IN_SE;
		df[nodeIndex + 6] = F_IN_NE;
	}

	else if(bc_type == WEST){
		
		nodeIndex = (0 + ty * LX) * 9;

		F_IN_W = df[nodeIndex + 2] + 2. / 3. * vx/vScale;
		F_IN_SW = df[nodeIndex + 6] + 1. / 6. * (vx + vy)/vScale;
		F_IN_NW = df[nodeIndex + 7] + 1. / 6. * (vx - vy)/vScale;

		df[nodeIndex + 1] = F_IN_W;
		df[nodeIndex + 5] = F_IN_SW;
		df[nodeIndex + 8] = F_IN_NW;
	}

	else if(bc_type == NORTH){
		
		nodeIndex = (tx + (LY - 1) * LX) * 9;
		
		F_IN_N = df[nodeIndex + 3] - 2. / 3. * vy/vScale;
		F_IN_NW = df[nodeIndex + 7] + 1. / 6. * (vx - vy)/vScale;
		F_IN_NE = df[nodeIndex + 5] - 1. / 6. * (vx + vy)/vScale;		

		df[nodeIndex + 4] = F_IN_N;
		df[nodeIndex + 8] = F_IN_NW;
		df[nodeIndex + 6] = F_IN_NE;
	}

	else if(bc_type == SOUTH){
	
		nodeIndex = tx * 9;

		F_IN_S = df[nodeIndex + 4] + 2. / 3. * vy/vScale;
		F_IN_SW = df[nodeIndex + 8] + 1. / 6. * (vx + vy)/vScale;
		F_IN_SE = df[nodeIndex + 6] - 1. / 6. * (vx - vy)/vScale;	
		
		df[nodeIndex + 3] = F_IN_S;
		df[nodeIndex + 7] = F_IN_SW;
		df[nodeIndex + 5] = F_IN_SE;
	}
  }
}
