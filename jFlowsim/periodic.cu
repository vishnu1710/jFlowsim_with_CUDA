#include "LB.h"

extern "C"

__global__ void PeriodicBC(float *df, int periodic,int LX, int LY)
{
//////////////////////////////////////////////////////////////////////////////////////////////////
	
	int tIdx = threadIdx.x;
	int tIdy = threadIdx.y;
	
	int tx = tIdx + blockIdx.x * blockDim.x;
	int ty = tIdy + blockIdx.y * blockDim.y;
//////////////////////////////////////////////////////////////////////////////////////////////////

	if(periodic == 0){
		
	    

	int nodeSouth = (tx + (0) * LX) * 9;
	int nodeNorth = (tx + (LY - 1) * LX) * 9;

	df[nodeSouth + 3] = df[nodeNorth + 3];
	df[nodeSouth + 5] = df[nodeNorth + 5];		
	df[nodeSouth + 7] = df[nodeNorth + 7];

	df[nodeNorth + 4] = df[nodeSouth + 4];
	df[nodeNorth + 8] = df[nodeSouth + 8];
	df[nodeNorth + 6] = df[nodeSouth + 6];
  }

	if(periodic == 1){

	   
	int nodeWest = (0 + ty * LX) * 9;
	int nodeEast = ((LX - 1) + ty * LX) * 9;

	df[nodeWest + 1] = df[nodeEast + 1];
	df[nodeWest + 5] = df[nodeEast + 5];
	df[nodeWest + 8] = df[nodeEast + 8];

	df[nodeEast + 2] = df[nodeWest + 2];
	df[nodeEast + 7] = df[nodeWest + 7];
	df[nodeEast + 6] = df[nodeWest + 6];

    }

}
