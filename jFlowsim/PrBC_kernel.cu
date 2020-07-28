#include "LB.h"

extern "C"
__device__ void getBGKEquilibrium(float rho, float vx, float vy, float *dfeq)
{
		
	dfeq[0] = (4.f/9.f) * rho * (1.0 - 1.5 * (vx * vx + vy * vy)); 
        dfeq[1] = (1.f/9.f) * rho * (1.0 + 3.0 * vx + 4.5 * vx * vx - 1.5 * (vx * vx + vy * vy));
        dfeq[3] = (1.f/9.f) * rho * (1.0 + 3.0 * vy + 4.5 * vy * vy - 1.5 * (vx * vx + vy * vy));
        dfeq[2] = (1.f/9.f) * rho * (1.0 - 3.0 * vx + 4.5 * vx * vx - 1.5 * (vx * vx + vy * vy));
        dfeq[4] = (1.f/9.f) * rho * (1.0 - 3.0 * vy + 4.5 * vy * vy - 1.5 * (vx * vx + vy * vy));
        dfeq[5] = (1.f/36.f) * rho * (1.0 + 3.0 * (vx + vy) + 4.5 * (vx + vy) * (vx + vy) - 1.5 * (vx * vx + vy * vy));
       	dfeq[7] = (1.f/36.f) * rho * (1.0 + 3.0 * (-vx + vy) + 4.5 * (-vx + vy) * (-vx + vy) - 1.5 * (vx * vx + vy * vy));
        dfeq[6] = (1.f/36.f) * rho * (1.0 - 3.0 * (vx + vy) + 4.5 * (vx + vy) * (vx + vy) - 1.5 * (vx * vx + vy * vy));
        dfeq[8] = (1.f/36.f) * rho * (1.0 - 3.0 * (-vx + vy) + 4.5 * (-vx + vy) * (-vx + vy) - 1.5 * (vx * vx + vy * vy));
}

extern "C"
__device__ float getVeloX(int x, int y, int LX, int LY, float *df)
{
	int nodeIndex = (y * LX + x) * 9;

	return (df[nodeIndex + 1] - df[nodeIndex + 2] + df[nodeIndex + 5] - df[nodeIndex + 6] + df[nodeIndex + 8] - df[nodeIndex + 7]);
} 

extern "C"
__device__ float getVeloY(int x, int y, int LX, int LY, float *df)
{
	int nodeIndex = (y * LX + x) * 9;

	return (df[nodeIndex + 3] - df[nodeIndex + 4] + df[nodeIndex + 5] - df[nodeIndex + 6] - df[nodeIndex + 8] + df[nodeIndex + 7]);
} 

extern "C"

__global__ void LBMPressureBC(int bc_type, float *df, int LX, int LY, float press, float *dfeq)
{
//////////////////////////////////////////////////////////////////////////////////////////////////
	
	int tIdx = threadIdx.x;
	int tIdy = threadIdx.y;
	
	int tx = tIdx + blockIdx.x * blockDim.x;
	int ty = tIdy + blockIdx.y * blockDim.y;

	float density;
	int nodeIndex, nodeIndex_;
	float vx,vy;
//////////////////////////////////////////////////////////////////////////////////////////////////
if(tx < LX && ty < LY)
{
	nodeIndex = (tx + ty * LX) * 9;

density = df[nodeIndex + 0] + df[nodeIndex + 1] + df[nodeIndex + 2] + df[nodeIndex + 3] + df[nodeIndex + 4] + df[nodeIndex + 5] + df[nodeIndex + 6] + df[nodeIndex + 7] + df[nodeIndex + 8];
	
	if(bc_type == EAST){
	
		vx = getVeloX(LX-1,ty, LX, LY, df) / density;
		vy = getVeloY(LX-1,ty, LX, LY, df) / density;

		getBGKEquilibrium(press,vx,vy,dfeq);

		nodeIndex_ = ((LX-1) + ty * LX) * 9;

		df[nodeIndex_ + 0] = dfeq[0];
		df[nodeIndex_ + 1] = dfeq[1];
		df[nodeIndex_ + 2] = dfeq[2];
		df[nodeIndex_ + 3] = dfeq[3];
		df[nodeIndex_ + 4] = dfeq[4];
		df[nodeIndex_ + 5] = dfeq[5];
		df[nodeIndex_ + 6] = dfeq[6];
		df[nodeIndex_ + 7] = dfeq[7];
		df[nodeIndex_ + 8] = dfeq[8];
	}
	

	else if(bc_type == WEST)
	{
		vx = getVeloX(0,ty, LX, LY, df) / density;
		vy = getVeloY(0,ty, LX, LY, df) / density;

		getBGKEquilibrium(press,vx,vy,dfeq);

		nodeIndex_ = (0 + ty * LX) * 9;

		df[nodeIndex_ + 0] = dfeq[0];
		df[nodeIndex_ + 1] = dfeq[1];
		df[nodeIndex_ + 2] = dfeq[2];
		df[nodeIndex_ + 3] = dfeq[3];
		df[nodeIndex_ + 4] = dfeq[4];
		df[nodeIndex_ + 5] = dfeq[5];
		df[nodeIndex_ + 6] = dfeq[6];
		df[nodeIndex_ + 7] = dfeq[7];
		df[nodeIndex_ + 8] = dfeq[8];
	}		



	else if(bc_type == NORTH)
	{
		vx = getVeloX(tx,LY-1, LX, LY, df) / density;
		vy = getVeloY(tx,LY-1, LX, LY, df) / density;

		getBGKEquilibrium(press,vx,vy,dfeq);

		nodeIndex_ = (tx + (LY-1) * LX) * 9;

		df[nodeIndex_ + 0] = dfeq[0];
		df[nodeIndex_ + 1] = dfeq[1];
		df[nodeIndex_ + 2] = dfeq[2];
		df[nodeIndex_ + 3] = dfeq[3];
		df[nodeIndex_ + 4] = dfeq[4];
		df[nodeIndex_ + 5] = dfeq[5];
		df[nodeIndex_ + 6] = dfeq[6];
		df[nodeIndex_ + 7] = dfeq[7];
		df[nodeIndex_ + 8] = dfeq[8];
	}		


	else if(bc_type == SOUTH)
	{
		vx = getVeloX(tx,0, LX, LY, df) / density;
		vy = getVeloY(tx,0, LX, LY, df) / density;

		getBGKEquilibrium(press,vx,vy,dfeq);

		nodeIndex_ = (tx + 0 * LX) * 9;

		df[nodeIndex_ + 0] = dfeq[0];
		df[nodeIndex_ + 1] = dfeq[1];
		df[nodeIndex_ + 2] = dfeq[2];
		df[nodeIndex_ + 3] = dfeq[3];
		df[nodeIndex_ + 4] = dfeq[4];
		df[nodeIndex_ + 5] = dfeq[5];
		df[nodeIndex_ + 6] = dfeq[6];
		df[nodeIndex_ + 7] = dfeq[7];
		df[nodeIndex_ + 8] = dfeq[8];
	}	
  }	
}
