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
__device__ float getDensity(int x, int y, int LX, int LY, float *df)
{
	int nodeIndex = (y * LX + x) * 9;
	return (df[nodeIndex + 0] + df[nodeIndex + 1] + df[nodeIndex + 2] + df[nodeIndex + 3] + df[nodeIndex + 4] + df[nodeIndex + 5] + df[nodeIndex + 6] + df[nodeIndex + 7] + df[nodeIndex + 8]);

}

extern "C"
__global__ void LBMVelocityBC(int bc_type, float *df, int LX, int LY, float vx, float vy, float *dfeq, float vScale)
{

//////////////////////////////////////////////////////////////////////////////////////////////////
	
	int tIdx = threadIdx.x;
	int tIdy = threadIdx.y;
	
	int tx = tIdx + blockIdx.x * blockDim.x;
	int ty = tIdy + blockIdx.y * blockDim.y;

	float density;
	int nodeIndex_;
	
//////////////////////////////////////////////////////////////////////////////////////////////////
if(tx < LX && ty < LY)
{

	if(bc_type == EAST){

	density = getDensity((LX-1),ty, LX, LY, df);
	getBGKEquilibrium(density, vx/vScale, vy/vScale, dfeq);

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

	else if(bc_type == WEST){

	density = getDensity(0,ty, LX, LY, df);
	getBGKEquilibrium(density, vx/vScale, vy/vScale, dfeq);

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

	else if(bc_type == NORTH){

	density = getDensity(tx,LY-1, LX, LY, df);
	getBGKEquilibrium(density, vx/vScale, vy/vScale, dfeq);

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

	else if(bc_type == SOUTH){

	density = getDensity(tx,0, LX, LY, df);
	getBGKEquilibrium(density, vx/vScale, vy/vScale, dfeq);

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
