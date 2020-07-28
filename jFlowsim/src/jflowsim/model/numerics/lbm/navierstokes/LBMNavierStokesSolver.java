package jflowsim.model.numerics.lbm.navierstokes;

import java.util.ArrayList;
import jflowsim.model.numerics.Solver;
import jflowsim.model.numerics.UniformGrid;
import jflowsim.model.numerics.lbm.LBMSolver;
import jflowsim.model.numerics.lbm.LBMSolverThread;
import jflowsim.model.numerics.lbm.LBMUniformGrid;
import java.util.concurrent.BrokenBarrierException;

import java.util.concurrent.CyclicBarrier;
import java.util.logging.Level;
import java.util.logging.Logger;


import static jcuda.driver.JCudaDriver.*;
import jcuda.*;
import jcuda.driver.*;
import jcuda.runtime.*;
import jcuda.utils.KernelLauncher;
import jflowsim.model.numerics.BoundaryCondition;
import jflowsim.model.numerics.lbm.LBMMovingWallBC;
import jflowsim.model.numerics.lbm.LBMPressureBC;
import jflowsim.model.numerics.lbm.LBMVelocityBC;

public class LBMNavierStokesSolver extends LBMSolver {

    public LBMNavierStokesSolver(UniformGrid grid) {
        super(grid);
    }

    @Override
    public void run() {

        switch(Solver.solverTyp){
            
            case "CPU":
                System.out.println(this.getClass().getSimpleName()+" num_of_threads:"+num_of_threads);
                //System.out.println("Thread here " + this.thread.getId());
        // init threads
        for (int rank = 0; rank < num_of_threads; rank++) {
            threadList.add(new LBMNavierStokesThread(rank, num_of_threads, this, grid, barrier));
        }

        // start threads and wait until all have finished
        startThreads();
                
        break;
        
            case "GPU":
                LBMGPUSolverThread cuda = new LBMGPUSolverThread();
                System.out.println("GPU Thread started ");
                cuda.cudaLunch(this);
                cuda.cudaFree();
                break;
        }
    }

    /* ======================================================================================= */
    class LBMNavierStokesThread extends LBMSolverThread {

        /* ======================================================================================= */
        public LBMNavierStokesThread(int myrank, int num_of_threads, Solver solver, LBMUniformGrid grid, CyclicBarrier barrier) {
            super(myrank, num_of_threads, solver, grid, barrier);
        }

        /* ======================================================================================= */
        public void run() {
            
            
                
            try {               

                long timer = 0;
                long counter = 0;
                

                if (myrank == 0) {
                    timer = System.currentTimeMillis();
                   
                }
                
                while (true) {
        
                    if (myrank == 0) {
                        grid.timestep++;
                        counter++;
                        
                    }
                    //System.out.println("Threads num " + myrank + '\t' + grid.timestep + '\t' + counter );
                    
                    double s_nu = 2. / (6. * grid.nue_lbm + 1.);

                    this.collisionFCM(s_nu);
                    //this.collisionMRT(s_nu);
                    //this.collision(s_nu);

                    this.addForcing();

////////////////////////////////////////////////////////////////////////////////
                   /* for(int k=0; k< grid.nx * grid.nx * 9; k++){
                System.out.println(" value is " + grid.f[k]);}*/
////////////////////////////////////////////////////////////////////////////////

                    
                    this.propagate();
                    
                    
                    //this.applyBCs();
                    if (myrank == 0) {
                        this.applyBCsNEW();
                         //System.out.println(" rank " + i++);
                    }
                    this.periodicBCs();
/////////////////////////////////////////////////////////////////////////////////
                    /*for(int k=0; k< grid.nx * grid.nx * 9; k++){
                System.out.println(" value is " + grid.f[k]);}*/
//////////////////////////////////////////////////////////////////////////////////                    
                    barrier.await();

                    if (myrank == 0) {
                        if (grid.timestep % grid.updateInterval == 0) {

                            long timespan = ((System.currentTimeMillis() - timer));

                            if (timespan > 0) {

                                grid.real_time = grid.timestep * grid.nue_lbm / grid.viscosity * Math.pow(grid.getLength() / grid.nx, 2.0);
                                grid.mnups = (grid.nx * grid.ny) * counter / timespan / 1000.0;

                                solver.update();
                            }

                            if ((System.currentTimeMillis() - timer) > 2000) {
                                timer = System.currentTimeMillis();
                                counter = 0;
                            }

                            //grid.adjustMachNumber(0.1);
                        }
                    }


                    // check if thread is interrupted
                    if (isInterrupted()) {
                        return;
                    }
                }
////////////////////////////////////////////////////////////////////////////////                
               /*if(myrank == 10){
                for(int k=0; k< grid.nx * grid.ny * 9; k++){
                System.out.println(" value is at " + k + " "+ grid.f[k]);
               }
               }*/
/////////////////////////////////////////////////////////////////////////////////                

            } catch (Exception ex) {
                //System.out.println("OVERALL EXCEPTION " + ex);
                return;
            }
            /*for(int k=0; k< grid.nx * grid.ny * 9; k++){
                System.out.println(" value is " + grid.f[k]);}*/
        }
        
    }//end: solveInrthread
    
    
    class LBMGPUSolverThread {
        
        protected int size, fsize;
        protected int LX,LY;
        
        //protected double [] feq;
        double s_nu = 2. / (6. * grid.nue_lbm + 1.);
        
        int num_threads = 256;
        int num_blocks_x, num_blocks_y;
        
        long timer = 0;
        long counter = 0;
        
        float elapsedtim[], elapsedtim_1[], feq[];
        int periodic = -1; 
        //public ArrayList<Integer> bc_type = new ArrayList<Integer>();
        //public ArrayList<String>bc_name = new ArrayList<>();
        String bc_name;
        int bc_type;
       
        CUdeviceptr df, dftemp, dtype, dfeq;
        
        KernelLauncher kernelLauncher1, kernelLauncher2,kernelLauncher3,kernelLauncher4,kernelLauncher5, kernelLauncher6;
        
        cudaEvent_t start, stop, start_1, stop_1;
        cudaStream_t stream;
        
        
        public LBMGPUSolverThread(){
            
            //grid.nx = 60; grid.ny = 11;
             LX = grid.nx;
             LY = grid.ny;
            
             size = LX * LY;
             fsize = size * 9;
             feq = new float[9];
             df = new CUdeviceptr();
             dftemp = new CUdeviceptr();
             dtype = new CUdeviceptr();
             dfeq = new CUdeviceptr();
             
             grid.hf = grid.toFloatArray(grid.f);
             grid.hftemp = grid.toFloatArray(grid.ftemp);
             
             start = new cudaEvent_t();
             stop = new cudaEvent_t();
             stream = new cudaStream_t();
             
             JCuda.cudaEventCreate(start);
             JCuda.cudaEventCreate(stop);
             
            
             
             elapsedtim = new float[1];
             //elapsedtim_1 = new float[1];
             
        }
        
        public void cudaLunch(Solver solver){
        System.out.println("Preparing the KernelLauncher...");
             
            kernelLauncher1 =
            KernelLauncher.load("lbm.ptx", "LBkernel");
            
            kernelLauncher2 =
            KernelLauncher.load("lbm_bc_noslip.ptx", "LBMNoSlipBC");
            
            kernelLauncher3 =
            KernelLauncher.load("lbm_bc_mw.ptx", "LBMMovingWallBC");
            
            kernelLauncher4 =
            KernelLauncher.load("lbm_bc_vel.ptx", "LBMVelocityBC");
                
            kernelLauncher5 =
            KernelLauncher.load("lbm_bc_pr.ptx", "LBMPressureBC");
            
            kernelLauncher6 =
            KernelLauncher.load("lbm_bc_per.ptx", "PeriodicBC");
            
        System.out.println("Initializing device memory........");
             
            
            cuMemAlloc(df, fsize * Sizeof.FLOAT);
            cuMemcpyHtoD(df, Pointer.to(grid.hf), fsize * Sizeof.FLOAT);
        
            
            cuMemAlloc(dftemp, fsize * Sizeof.FLOAT);
            cuMemcpyHtoD(dftemp, Pointer.to(grid.hftemp), fsize * Sizeof.FLOAT);
            
            cuMemAlloc(dfeq, 9 * Sizeof.FLOAT);
            cuMemcpyHtoD(dfeq, Pointer.to(feq), 9 * Sizeof.FLOAT);
            
           /* CUdeviceptr dfeq = new CUdeviceptr();
            cuMemAlloc(dfeq, fsize * Sizeof.DOUBLE);
            cuMemcpyHtoD(dfeq, Pointer.to(grid.f), fsize * Sizeof.DOUBLE);*/
            
           /*cuMemAlloc(df, fsize * Sizeof.FLOAT);
            cuMemcpyHtoD(df, Pointer.to(grid.f), fsize * Sizeof.FLOAT);
        
            
            cuMemAlloc(dftemp, fsize * Sizeof.FLOAT);
            cuMemcpyHtoD(dftemp, Pointer.to(grid.ftemp), fsize * Sizeof.FLOAT);*/
           
           
            
            cuMemAlloc(dtype, size * Sizeof.INT);
            cuMemcpyHtoD(dtype, Pointer.to(grid.type), size * Sizeof.INT);
            
            if(grid.periodicX)
                 periodic = 1;
            else if(grid.periodicY)
                periodic = 0;
                                             
            
            System.out.println("Kernel Launch...");
            
            num_blocks_x = Math.round(grid.nx / num_threads) + 1;
            
            //num_blocks_y = Math.round(grid.ny / num_threads) + 1;
            //System.out.println("blocks " + num_blocks_x);
            kernelLauncher1.setGridSize(num_blocks_x, grid.ny);
            kernelLauncher1.setBlockSize(num_threads, 1, 1);
            
            kernelLauncher2.setGridSize(num_blocks_x, grid.ny);
            kernelLauncher2.setBlockSize(num_threads, 1, 1);
            
            kernelLauncher3.setGridSize(num_blocks_x, grid.ny);
            kernelLauncher3.setBlockSize(num_threads, 1, 1);
            
            kernelLauncher4.setGridSize(num_blocks_x, grid.ny);
            kernelLauncher4.setBlockSize(num_threads, 1, 1);
            
            kernelLauncher5.setGridSize(num_blocks_x, grid.ny);
            kernelLauncher5.setBlockSize(num_threads, 1, 1);
            
            /*kernelLauncher6.setGridSize(num_blocks_x, grid.ny);
            kernelLauncher6.setBlockSize(num_threads, 1, 1);*/
            
            //timer = System.currentTimeMillis();
            JCuda.cudaEventRecord(start, stream);
            
            while(true){
                     
            
            kernelLauncher1.call(LX, LY, df, dtype, (float)s_nu, (float)grid.forcingX1, (float)grid.forcingX2, periodic);
                
               
            cudaBC();
            
            /*if(periodic != -1)
            kernelLauncher6.call(df, periodic, LX, LY);*/
            
            
                       
            
            grid.timestep++;
            counter+=1;
            
                        if (grid.timestep % grid.updateInterval == 0) {

                            JCuda.cudaEventRecord(stop, stream);
                            JCuda.cudaEventSynchronize(stop);
                            JCuda.cudaEventElapsedTime(elapsedtim, start, stop);
                            //long timespan = ((System.currentTimeMillis() - timer));
                            
                            long timespan = (long)elapsedtim[0];

                            if (timespan > 0) {

                                grid.real_time = grid.timestep * grid.nue_lbm / grid.viscosity * Math.pow(grid.getLength() / grid.nx, 2.0);
                                grid.mnups = (grid.nx * grid.ny) * counter / timespan / 1000.0;

                                solver.update();
                            }
                            
                            
                            JCuda.cudaEventRecord(stop, stream);
                            JCuda.cudaEventSynchronize(stop);
                            JCuda.cudaEventElapsedTime(elapsedtim, start, stop);
                            
                            //if ((System.currentTimeMillis() - timer) > 2000) {
                            if((long)elapsedtim[0] > 2000){
                                JCuda.cudaEventRecord(start, stream);
                                //timer = System.currentTimeMillis();
                                counter = 0;
                            }
            }
                        
                                                    
            cuMemcpyDtoH(Pointer.to(grid.hf), df, fsize * Sizeof.FLOAT);
            grid.f = grid.toDoubleArray(grid.hf);
            cuMemcpyHtoD(dtype, Pointer.to(grid.type), size * Sizeof.INT);
                        
            
            if(thread.isInterrupted()){
               return;
            }
            
        }
            
            
         
    }
        
        public void cudaBC(){
            for (BoundaryCondition bc : grid.bcList) {
                    
                    bc_name = (bc.getClass().getSimpleName());
                    
                    switch(bc_name){
                        
                        case "LBMNoSlipBC":
                            kernelLauncher2.call(bc.getType(), df, LX, LY);
                            break;
                            
                        case "LBMMovingWallBC":
                            kernelLauncher3.call(bc.getType(), df, LX, LY, LBMMovingWallBC.getVelX(), LBMMovingWallBC.getVelY(),(float)grid.dv);
                            
                            break;
                            
                        case "LBMVelocityBC":
                            kernelLauncher4.call(bc.getType(), df, LX, LY, LBMVelocityBC.getVelX(), LBMVelocityBC.getVelY(), dfeq, (float)grid.dv);
                            //System.out.println("velocity of" + LBMVelocityBC.getVelX());
                            break;
                            
                        case "LBMPressureBC":
                            kernelLauncher5.call(bc.getType(), df, LX, LY, LBMPressureBC.getPress(),dfeq);
                            break;
                          
                    }
            }
        }
        
        public void cudaFree(){
            // Clean up
            cuMemFree(df);
            cuMemFree(dfeq);
            cuMemFree(dtype);
            JCuda.cudaEventDestroy(start);
            JCuda.cudaEventDestroy(stop);
            System.out.println("Device pointer cleaned.......\n");
        }
    }
        
        
    
    
}//end:solver

