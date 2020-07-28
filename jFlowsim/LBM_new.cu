#include "LB.h"

extern "C"
__device__ double getForcingForDirection(int dir, double fX1, double fX2)
{
	switch(dir)
	{
		case 0:
			return 0;
			
		case 1:
			return ((1.f/3.f) * fX1);
			
				
		case 2:
			return ((-1.f/3.f) * fX1);
						
			
		case 3:
			return ((1.f/3.f) * fX2);
			
			
		case 4:
			return ((-1.f/3.f) * fX2);
					
			
		case 5:
			return ((1.f/12.f) * (fX1 + fX2));
			
			
		case 7:
			return ((1.f/12.f) * (-fX1 + fX2));
			
			
		case 6:
			return ((1.f/12.f) * (-fX1 - fX2));
			
			
		case 8:
			return ((1.f/12.f) * (fX1 - fX2));
			
			
		default:
			return -999.9;
			
			
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
__global__ void LBMNoSlipBC(int bc_type, double *df,double *dftemp, int LX, int LY)
{

//////////////////////////////////////////////////////////////////////////////////////////////////
	int num_threads = blockDim.x;
	int tIdx = threadIdx.x;
	int tIdy = threadIdx.y;
	
	int tx = tIdx + blockIdx.x * blockDim.x;
	int ty = tIdy + blockIdx.y * blockDim.y;
//////////////////////////////////////////////////////////////////////////////////////////////////
if(tx < LX && ty < LY)
{
	int nodeIndex;

	if(bc_type == EAST){
		
		nodeIndex = ((LX -1) + ty * LX) * 9;
		
		df[nodeIndex + 2] = dftemp[nodeIndex + 1];
		df[nodeIndex + 7] = dftemp[nodeIndex + 8];
		df[nodeIndex + 6] = dftemp[nodeIndex + 5];
	}
	else if(bc_type == WEST){
		
		nodeIndex = (0 + ty * LX) * 9;

		df[nodeIndex + 1] = dftemp[nodeIndex + 2];
		df[nodeIndex + 5] = dftemp[nodeIndex + 6];
		df[nodeIndex + 8] = dftemp[nodeIndex + 7];
	}
	else if(bc_type == NORTH){
		
		nodeIndex = (tx + (LY - 1) * LX) * 9;
		
		df[nodeIndex + 4] = dftemp[nodeIndex + 3];
		df[nodeIndex + 8] = dftemp[nodeIndex + 7];
		df[nodeIndex + 6] = dftemp[nodeIndex + 5];
	}
	else if(bc_type == SOUTH){
	
		nodeIndex = tx * 9;
		
		df[nodeIndex + 3] = dftemp[nodeIndex + 4];
		df[nodeIndex + 7] = dftemp[nodeIndex + 8];
		df[nodeIndex + 5] = dftemp[nodeIndex + 6];
	}
}
}


	


extern "C"

__global__ void LBMkernel(int LX, int LY, double *df,  int *dtype, double s_nu, double fX1, double fX2, int periodic, int bc_type)
{

//////////////////////////////////////////////////////////////////////////////////////////////////
	int num_threads = blockDim.x;
	int tIdx = threadIdx.x;
	int tIdy = threadIdx.y;
	
	int tx = tIdx + blockIdx.x * blockDim.x;
	int ty = tIdy + blockIdx.y * blockDim.y;
//////////////////////////////////////////////////////////////////////////////////////////////////
	
	int offset, nodeIndex, offset_, nodeIndex_;
	/*int invdir[9] = {0, 2, 1, 4, 3, 6, 5, 8, 7};
	int ex[] = {0, 1, -1, 0, 0, 1, -1, -1, 1};
	int ey[] = {0, 0, 0, 1, -1, 1, -1, 1, -1};*/

	double rho, vx, vy, tmp;
	

	double P, NE, V, kxxyy, UP, RIGHT, NP;

	double F_IN_R, F_IN_E, F_IN_N, F_IN_W, F_IN_S, F_IN_NE, F_IN_NW, F_IN_SW, F_IN_SE;

   __shared__ double F_OUT_E[THREAD_NUM];
   __shared__ double F_OUT_W[THREAD_NUM];
   __shared__ double F_OUT_NE[THREAD_NUM];
   __shared__ double F_OUT_NW[THREAD_NUM];
   __shared__ double F_OUT_SW[THREAD_NUM];
   __shared__ double F_OUT_SE[THREAD_NUM];

	//BC bc_name = bcname;

if(tx < LX && ty < LY)
{

	offset = tx + ty * LX;
	nodeIndex = offset * 9;
	
	F_IN_R = df[nodeIndex + 0];
	F_IN_E = df[nodeIndex + 1];
	F_IN_W = df[nodeIndex + 2];
	F_IN_N = df[nodeIndex + 3];
	F_IN_S = df[nodeIndex + 4];
	F_IN_NE = df[nodeIndex + 5];
	F_IN_NW = df[nodeIndex + 7];
	F_IN_SE = df[nodeIndex + 8];
	F_IN_SW = df[nodeIndex + 6];


   if(dtype[offset] == FLUID){
	
	rho = ((F_IN_NW + F_IN_SW) + (F_IN_SE + F_IN_NE)) + ((F_IN_W + F_IN_S) + (F_IN_E + F_IN_N)) + F_IN_R;

	vx = ((F_IN_NE - F_IN_SW) + (F_IN_SE - F_IN_NW)) + (F_IN_E - F_IN_W);
	vy = ((F_IN_NE - F_IN_SW) + (F_IN_NW - F_IN_SE)) + (F_IN_N - F_IN_S);

	vx/=rho;
	vy/=rho;

	P = (1./12.*(rho*(vx*vx+vy*vy)-F_IN_E-F_IN_N-F_IN_S-F_IN_W-2*(F_IN_SE+F_IN_SW+F_IN_NE+F_IN_NW-1./3.*rho)));
	NE = (s_nu*0.25*(F_IN_N+F_IN_S-F_IN_E-F_IN_W+rho*(vx*vx-vy*vy)));
	V = (s_nu*((F_IN_NE+F_IN_SW-F_IN_NW-F_IN_SE)-vx*vy*rho)*0.25);

	kxxyy = (F_IN_E+F_IN_NE+F_IN_NW+F_IN_SE+F_IN_SW+F_IN_W-vx*vx*rho+2*NE+6*P)/rho*(F_IN_N+F_IN_NE+F_IN_NW+F_IN_S+F_IN_SE+F_IN_SW-vy*vy*rho-2*NE+6*P)/rho;

	UP = (-(.25*(F_IN_SE+F_IN_SW-F_IN_NE-F_IN_NW-2.*vx*vx*vy*rho+vy*(rho-F_IN_N-F_IN_S-F_IN_R))-vy*.5*(-3.*P-NE)+vx*((F_IN_NE-F_IN_NW-F_IN_SE+F_IN_SW)*.5-2*V)));

	RIGHT = (-(.25*(F_IN_SW+F_IN_NW-F_IN_SE-F_IN_NE-2.*vy*vy*vx*rho+vx*(rho-F_IN_R-F_IN_W-F_IN_E))-vx*.5*(-3.*P+NE)+vy*((F_IN_NE+F_IN_SW-F_IN_SE-F_IN_NW)*.5-2*V)));

	 NP = (0.25*(rho * (kxxyy) -F_IN_NE-F_IN_NW-F_IN_SE-F_IN_SW-8*P+2*(vx*(F_IN_NE-F_IN_NW+F_IN_SE-F_IN_SW-4*RIGHT)+vy*(F_IN_NE+F_IN_NW-F_IN_SE-F_IN_SW-4*UP))
         +4*vx*vy*(-F_IN_NE+F_IN_NW+F_IN_SE-F_IN_SW+4*V) +vx*vx*(-F_IN_N-F_IN_NE-F_IN_NW-F_IN_S-F_IN_SE-F_IN_SW+2*NE-6*P) 
         +vy*vy*((-F_IN_E-F_IN_NE-F_IN_NW-F_IN_SE-F_IN_SW-F_IN_W-2*NE-6*P)+3*vx*vx*rho)));


      F_IN_NW = F_IN_NW + 2 * P + NP + V - UP + RIGHT;
      F_IN_W = F_IN_W  - P - 2 * NP + NE - 2 * RIGHT;
      F_IN_SW = F_IN_SW + 2 * P + NP - V + UP + RIGHT;
      F_IN_S = F_IN_S - P - 2 * NP - NE - 2 * UP;
      F_IN_SE = F_IN_SE + 2 * P + NP + V + UP - RIGHT;
      F_IN_E = F_IN_E  - P - 2 * NP + NE + 2 * RIGHT;
      F_IN_NE = F_IN_NE + 2 * P + NP - V - UP - RIGHT;
      F_IN_N = F_IN_N - P - 2 * NP - NE + 2 * UP;
      F_IN_R = F_IN_R + (4 * (-P + NP));
  }

  else if(dtype[offset] == SOLID){

	/*for (int dir = 0; dir < 9; dir++) {
                        dftemp[nodeIndex + dir] = df[nodeIndex + invdir[dir]];
                    }*/

     tmp=F_IN_E ; F_IN_E =F_IN_W ; F_IN_W =tmp;
      tmp=F_IN_N ; F_IN_N =F_IN_S ; F_IN_S =tmp;
      tmp=F_IN_NE ; F_IN_NE =F_IN_SW ; F_IN_SW =tmp;
      tmp=F_IN_NW ; F_IN_NW =F_IN_SE ; F_IN_SE =tmp;
   }

//Adding force


	/*for (int dir = 0; dir < 9; dir++) {
                    dftemp[nodeIndex + dir] += getForcingForDirection(dir, fX1, fX2);
                }*/

	F_IN_R += getForcingForDirection(0, fX1, fX2);
	F_IN_E += getForcingForDirection(1, fX1, fX2);	
	F_IN_W += getForcingForDirection(2, fX1, fX2);
	F_IN_N += getForcingForDirection(3, fX1, fX2);
	F_IN_S += getForcingForDirection(4, fX1, fX2);
	F_IN_NE += getForcingForDirection(5, fX1, fX2);
	F_IN_SE += getForcingForDirection(8, fX1, fX2);
	F_IN_SW += getForcingForDirection(6, fX1, fX2);
	F_IN_NW += getForcingForDirection(7, fX1, fX2);
			
	
	/*df[nodeIndex + 0] = dftemp[nodeIndex + 0];                  
	df[nodeIndex + 1] = dftemp[nodeIndex + 1];
	df[nodeIndex + 2] = dftemp[nodeIndex + 2];
	df[nodeIndex + 3] = dftemp[nodeIndex + 3];
	df[nodeIndex + 4] = dftemp[nodeIndex + 4];
	df[nodeIndex + 5] = dftemp[nodeIndex + 5];
	df[nodeIndex + 7] = dftemp[nodeIndex + 7];
	df[nodeIndex + 8] = dftemp[nodeIndex + 8];
	df[nodeIndex + 6] = dftemp[nodeIndex + 6];*/


	/*F_IN_R = dftemp[nodeIndex + 0];
	F_IN_E = dftemp[nodeIndex + 1];
	F_IN_W = dftemp[nodeIndex + 2];
	F_IN_N = dftemp[nodeIndex + 3];
	F_IN_S = dftemp[nodeIndex + 4];
	F_IN_NE = dftemp[nodeIndex + 5];
	F_IN_NW = dftemp[nodeIndex + 7];
	F_IN_SE = dftemp[nodeIndex + 8];
	F_IN_SW = dftemp[nodeIndex + 6];*/


//Propagation

	//df[nodeIndex + 0] = dftemp[nodeIndex + 0];
	//df[nodeIndex + 3] = F_IN_N;
	//df[nodeIndex + 4] = F_IN_S;

	/*if(tIdy > 0){
		offset_ = (ex[4] + ey[4] * LX) * 9;
		df[(nodeIndex + 4) + offset_] = dftemp[nodeIndex + 4];
		}
		
	if(tIdy < LY-1){
		offset_ = (ex[3] + ey[3] * LX) * 9;
		df[(nodeIndex + 3) + offset_] = dftemp[nodeIndex + 3];
		}

	//E propagation in shared memory
		if(tIdx < LX-1){
			F_OUT_E[tIdx+1] = dftemp[nodeIndex + 1];
			F_OUT_NE[tIdx+1] = dftemp[nodeIndex + 5];
			F_OUT_SE[tIdx+1] = dftemp[nodeIndex + 8];
		}
		else if(tIdx > 0){
			F_OUT_W[tIdx-1] = dftemp[nodeIndex + 2];
			F_OUT_NW[tIdx-1] = dftemp[nodeIndex + 7];
			F_OUT_SW[tIdx-1] = dftemp[nodeIndex + 6];
		}

	__syncthreads();

	if(tIdx > 0){
		df[nodeIndex + 1] = F_OUT_E[tIdx];
		//if(tIdy > 0)
			df[nodeIndex + 8] = F_OUT_SE[tIdx];
		//if(tIdy < LY-1)
			df[nodeIndex + 5] = F_OUT_NE[tIdx];
	}

	if(tIdx < LX-1){
		df[nodeIndex + 2] = F_OUT_W[tIdx];
		//if(tIdy > 0)
			df[nodeIndex + 6] = F_OUT_SW[tIdx];
		//if(tIdy < LY-1)
			df[nodeIndex + 7] = F_OUT_NW[tIdx];
	}*/

	
	
	

 if(tx==0){
      F_OUT_E [tIdx+1] = F_IN_E;
      F_OUT_NE[tIdx+1] = F_IN_NE;
      F_OUT_SE[tIdx+1] = F_IN_SE;

      
      F_OUT_W[LX-1] = F_IN_W;
      F_OUT_NW[LX-1] = F_IN_NW;
      F_OUT_SW[LX-1] = F_IN_SW;
     

   }
   else if(tx == LX -1){
	
      F_OUT_E [0] = F_IN_E;
      F_OUT_NE[0] = F_IN_NE;
      F_OUT_SE[0] = F_IN_SE;

	
      F_OUT_W [tIdx-1]=F_IN_W;
      F_OUT_NW[tIdx-1]=F_IN_NW;
      F_OUT_SW[tIdx-1]=F_IN_SW;
   }
   else {
	//printf("thread id %d\n",tIdx);
      F_OUT_E [tIdx+1]=F_IN_E;
      F_OUT_NE[tIdx+1]=F_IN_NE;
      F_OUT_SE[tIdx+1]=F_IN_SE;
      F_OUT_W [tIdx-1]=F_IN_W;
      F_OUT_NW[tIdx-1]=F_IN_NW;
      F_OUT_SW[tIdx-1]=F_IN_SW;
   }

   __syncthreads();

	 df[nodeIndex + 0] = F_IN_R;
	
	 df[nodeIndex + 1] = F_OUT_E[tIdx];
   	 df[nodeIndex + 2] = F_OUT_W[tIdx];
	
	if(ty < LY-1){
   	offset_ = tx + (ty + 1) * LX;
	nodeIndex_ = offset_ * 9;
	
	 df[nodeIndex_ + 3] = F_IN_N;
	 df[nodeIndex_ + 5] = F_OUT_NE[tIdx];
   	 df[nodeIndex_ + 7] = F_OUT_NW[tIdx];	
	}

	if(ty > 0){
	offset_ = tx + (ty - 1) * LX;
	nodeIndex_ = offset_ * 9;

	 df[nodeIndex_ + 4] = F_IN_S;
	 df[nodeIndex_ + 8] = F_OUT_SE[tIdx];
   	 df[nodeIndex_ + 6] = F_OUT_SW[tIdx];

	}
	
	/*for(int dir = 0; dir < 9; dir++)
	{	
		offset_ = (ex[dir] + ey[dir] * LX) * 9;
		nodeIndex_ = nodeIndex + dir;
		
		df[nodeIndex_ + offset_] = dftemp[nodeIndex_];
	}*/
  	
		
	periodicBCs(df,tx, ty,periodic, LX, LY);
   
	}

}//global end
