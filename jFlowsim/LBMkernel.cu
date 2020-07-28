#include "LB.h"

extern "C"
__device__ void getBGKEquilibrium(double rho, double vx, double vy, double *dfeq, int nodeIndex)
{
		
	dfeq[nodeIndex + 0] = (4.f/9.f) * rho * (1.0 - 1.5 * (vx * vx + vy * vy)); 
        dfeq[nodeIndex + 1] = (1.f/9.f) * rho * (1.0 + 3.0 * vx + 4.5 * vx * vx - 1.5 * (vx * vx + vy * vy));
        dfeq[nodeIndex + 3] = (1.f/9.f) * rho * (1.0 + 3.0 * vy + 4.5 * vy * vy - 1.5 * (vx * vx + vy * vy));
        dfeq[nodeIndex + 2] = (1.f/9.f) * rho * (1.0 - 3.0 * vx + 4.5 * vx * vx - 1.5 * (vx * vx + vy * vy));
        dfeq[nodeIndex + 4] = (1.f/9.f) * rho * (1.0 - 3.0 * vy + 4.5 * vy * vy - 1.5 * (vx * vx + vy * vy));
        dfeq[nodeIndex + 5] = (1.f/36.f) * rho * (1.0 + 3.0 * (vx + vy) + 4.5 * (vx + vy) * (vx + vy) - 1.5 * (vx * vx + vy * vy));
       	dfeq[nodeIndex + 7] = (1.f/36.f) * rho * (1.0 + 3.0 * (-vx + vy) + 4.5 * (-vx + vy) * (-vx + vy) - 1.5 * (vx * vx + vy * vy));
        dfeq[nodeIndex + 6] = (1.f/36.f) * rho * (1.0 - 3.0 * (vx + vy) + 4.5 * (vx + vy) * (vx + vy) - 1.5 * (vx * vx + vy * vy));
        dfeq[nodeIndex + 8] = (1.f/36.f) * rho * (1.0 - 3.0 * (-vx + vy) + 4.5 * (-vx + vy) * (-vx + vy) - 1.5 * (vx * vx + vy * vy));
}


extern "C"
__device__ double getForcingForDirection(int dir, double fX1, double fX2, double res)
{
	switch(dir)
	{
		case 0:
			res = 0;
			return res;
			
		case 1:
			res = ((1.f/3.f) * fX1);
			return res;
				
		case 2:
			res = (-1.f/3.f) * fX1;
			return res;			
			
		case 3:
			res = (1.f/3.f) * fX2;
			return res;
			
		case 4:
			res = (-1.f/3.f) * fX2;
			return res;		
			
		case 5:
			res = (1.f/12.f) * (fX1 + fX2);
			return res;
			
		case 7:
			res = (1.f/12.f) * (-fX1 + fX2);
			return res;
			
		case 6:
			res = (1.f/12.f) * (-fX1 - fX2);
			return res;
			
		case 8:
			res = (1.f/12.f) * (fX1 - fX2);
			return res;
			
		default:
			res = -999.9;
			return res;
			
	}
}

extern "C"
	//PeriodicBC's

 __device__ void periodicBCs(double *df, int tx, int ty, int periodic,int LX, int LY)
{
	if(periodic == 0){
		
	    int nodeNorth = 0;
            int nodeSouth = 0;

	nodeSouth = (tx + (0) * LX) * 9;
	nodeNorth = (tx + (LY - 1) * LX) * 9;

	df[nodeSouth + 3] = df[nodeNorth + 3];
	df[nodeSouth + 5] = df[nodeNorth + 5];		
	df[nodeSouth + 7] = df[nodeNorth + 7];

	df[nodeNorth + 4] = df[nodeSouth + 4];
	df[nodeNorth + 8] = df[nodeSouth + 8];
	df[nodeNorth + 6] = df[nodeSouth + 6];
     }

	if(periodic == 1){

	    int nodeEast = 0;
            int nodeWest = 0;

	nodeWest = (0 + ty * LX) * 9;
	nodeEast = ((LX - 1) + ty * LX) * 9;

	df[nodeWest + 1] = df[nodeEast + 1];
	df[nodeWest + 5] = df[nodeEast + 5];
	df[nodeWest + 8] = df[nodeEast + 8];

	df[nodeEast + 2] = df[nodeWest + 2];
	df[nodeEast + 7] = df[nodeWest + 7];
	df[nodeEast + 6] = df[nodeWest + 6];

    }

}

extern "C"

__global__ void LBkernel(int LX, int LY, double *df, double *dfeq, double *dftemp, int *dtype, double s_nu, double fX1, double fX2, int periodic)
{
	int offset, nodeIndex, offset_, nodeIndex_, jn, jp, in, ip;
	int num_threads = THREAD_NUM;
	int invdir[9] = {0, 2, 1, 4, 3, 6, 5, 8, 7};//{0,3,4,1,2,7,8,5,6};

	int ex[] = {0, 1, -1, 0, 0, 1, -1, -1, 1};	
	int ey[] = {0, 0, 0, 1, -1, 1, -1, 1, -1};

	double rho, vx, vy, tmp;
	double res;
	//double P, NE, V, kxxyy, UP, NP, RIGHT;
	//double R, E, W, S, N, Ne, Sw, Nw, Se;
	
   __shared__ double F_OUT_E[THREAD_NUM];
   __shared__ double F_OUT_W[THREAD_NUM];
   //__shared__ double F_OUT_N[THREAD_NUM];
   //__shared__ double F_OUT_S[THREAD_NUM];
   __shared__ double F_OUT_NE[THREAD_NUM];
   __shared__ double F_OUT_NW[THREAD_NUM];
   __shared__ double F_OUT_SW[THREAD_NUM];
   __shared__ double F_OUT_SE[THREAD_NUM];

	int tIdx = threadIdx.x;
	int tIdy= threadIdx.y;
	int tx = tIdx + blockIdx.x * blockDim.x;
	int ty = tIdy + blockIdx.y * blockDim.y;
	int tot_x = (gridDim.x * blockDim.x);
	
	//offset_glob = tx + ty * (gridDim.x * blockDim.x);
	
	
  if(tx < LX && ty < LY ){

	offset = tx + ty * LX;
	

	if(dtype[offset] == FLUID)
    {
	//offset = tx + ty * LX;
	//printf("tx is %d\n", SOLID);
	
	nodeIndex = offset * 9;		

	rho = df[nodeIndex + 0] + df[nodeIndex + 1] + df[nodeIndex + 2] + df[nodeIndex + 3] + df[nodeIndex + 4] + df[nodeIndex + 5] + 				df[nodeIndex + 6] + df[nodeIndex +7] + df[nodeIndex + 8];
	
	vx = (df[nodeIndex + 1] - df[nodeIndex +2] + df[nodeIndex + 5] - df[nodeIndex + 6] + df[nodeIndex + 8] - df[nodeIndex + 7]) / rho;

	vy =  (df[nodeIndex + 3] - df[nodeIndex +4] + df[nodeIndex + 5] - df[nodeIndex + 6] - df[nodeIndex + 8] + df[nodeIndex + 7]) / rho;

	getBGKEquilibrium(rho, vx, vy, dfeq, nodeIndex);

	for (int k=0; k<9; k++)
		{
			dftemp[nodeIndex + k] = df[nodeIndex + k] - (s_nu * (df[nodeIndex + k] - dfeq[nodeIndex + k]));
		}
	//printf("offset is %d\n", offset);
	
   }
 
   else if(dtype[offset] == SOLID)
	{
		nodeIndex = offset * 9;		
		
		for (int k=0; k<9; k++)
		{

			dftemp[nodeIndex + k] = df[nodeIndex + invdir[k]];	
		}
	}

//}

	//add forcing
	
	//if(tx < LX && ty < LY)
	//{
		//offset = tx + ty * LX;
		nodeIndex = offset * 9;
	
	for (int k=1; k<9; k++)
		{
			//getForcingForDirection(dir, fX1, fX2, res);
			dftemp[nodeIndex + k] += getForcingForDirection(k, fX1, fX2, res);;	
		}
	//}


	//Propagate

	//printf("offset is %d\n", (tx + ty * LX));
	

	//if(tx < LX && ty < LY)
	//{	
		//offset = tx + ty * LX;
		nodeIndex = offset * 9;
		
		if(tx == 0)
		{
			F_OUT_E [tIdx+1] = dftemp[nodeIndex + 1];
      			F_OUT_NE[tIdx+1] = dftemp[nodeIndex + 5];
			F_OUT_SE[tIdx+1] = dftemp[nodeIndex + 8];

			F_OUT_W [num_threads-1] = dftemp[nodeIndex + 2];
      			F_OUT_NW[num_threads-1] = dftemp[nodeIndex + 7];
      			F_OUT_SW[num_threads-1] = dftemp[nodeIndex + 6];

   		}

		else if(tx == LX - 1)
		{
			F_OUT_E [0] = dftemp[nodeIndex + 1];
      			F_OUT_NE[0] = dftemp[nodeIndex + 5];
      			F_OUT_SE[0] = dftemp[nodeIndex + 8];

      			F_OUT_W [tIdx-1] = dftemp[nodeIndex + 2];
      			F_OUT_NW[tIdx-1] = dftemp[nodeIndex + 7];
      			F_OUT_SW[tIdx-1] = dftemp[nodeIndex + 6];
   		}

		else if(tIdx > 0 && tIdx < num_threads-1){
			F_OUT_E [tIdx+1] = dftemp[nodeIndex + 1];
      			F_OUT_NE[tIdx+1] = dftemp[nodeIndex + 5];
      			F_OUT_SE[tIdx+1] = dftemp[nodeIndex + 8];
      			F_OUT_W [tIdx-1] = dftemp[nodeIndex + 2];
      			F_OUT_NW[tIdx-1] = dftemp[nodeIndex + 7];
      			F_OUT_SW[tIdx-1] = dftemp[nodeIndex + 6];
   		}

	__syncthreads();

	 df[nodeIndex + 0] = dftemp[nodeIndex + 0];
	 df[nodeIndex + 1] = F_OUT_E[tIdx];
   	 df[nodeIndex + 2] = F_OUT_W[tIdx];
	
	 offset_ = tx + (ty + 1) * LX;
	 nodeIndex_ = offset_ * 9;
	
	 df[nodeIndex_ + 3] = dftemp[nodeIndex + 3];
	 df[nodeIndex_ + 5] = F_OUT_NE[tIdx];
   	 df[nodeIndex_ + 7] = F_OUT_NW[tIdx];	


	 offset_ = tx + (ty - 1) * LX;
	 nodeIndex_ = offset_ * 9;

	 df[nodeIndex_ + 4] = dftemp[nodeIndex + 4];
	 df[nodeIndex_ + 8] = F_OUT_SE[tIdx];
   	 df[nodeIndex_ + 6] = F_OUT_SW[tIdx];

	//}


	/*//offset = tx + ty * LX;
			
	
	 jn = (ty>0)?(ty-1):(LY-1);
	
	//printf("threadId is %d\n", tIdx);
	
	 jp = (ty<LY-1)?(ty+1):(0);
	
	 in = (tx>0)?(tx-1):(LX-1); 
	 ip = (tx<LX-1)?(tx+1):(0);
	
	
			
		df[(tx + ty*LX)*9 + 0] = dftemp[(tx + ty*LX)*9 + 0]; 
		df[(ip + ty*LX)*9 + 1] = dftemp[(tx + ty*LX)*9 + 1]; 
		df[(tx + jp*LX)*9 + 3] = dftemp[(tx + ty*LX)*9 + 3];
		df[(in + ty*LX)*9 + 2] = dftemp[(tx + ty*LX)*9 + 2];
		df[(tx + jn*LX)*9 + 4] = dftemp[(tx + ty*LX)*9 + 4];
		df[(ip + jp*LX)*9 + 5] = dftemp[(tx + ty*LX)*9 + 5];
		df[(in + jp*LX)*9 + 7] = dftemp[(tx + ty*LX)*9 + 7];
		df[(in + jn*LX)*9 + 6] = dftemp[(tx + ty*LX)*9 + 6];
		df[(ip + jn*LX)*9 + 8] = dftemp[(tx + ty*LX)*9 + 8]; 


	
	//__syncthreads();
		}*/	//printf("count is %d\n",count);
	}

	periodicBCs(df,tx, ty,periodic, LX, LY);
}


