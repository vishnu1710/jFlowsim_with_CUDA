#ifndef _LB_H_
#define _LB_H_

#include<string.h> 

/*typedef enum{
LBMNoSlipBC,
LBMMovingWallBC,
LBMBounceForwardBC,
LBMPressureBC,
LBMVelocityBC,
}BC;*/

//typedef unsigned char BC;
#define BOUNDARY -2
#define SOLID  -1
#define INTERFACE 1
#define FLUID  2
#define INFLOWH 3
#define RUNUP 4
#define GAS 5
#define SOLIDN 6
#define SOLIDS 7
#define SOLIDE 8
#define SOLIDW 9
#define VELO 10
#define PRESS 11


//Boundary

#define VOID 0
#define EAST 1
#define WEST 2
#define NORTH 3
#define SOUTH 4

#define THREAD_NUM 256




#endif
