# jFlowsim_with_CUDA
jflowsim which was previously thread parallel in CPU mode (by ChristianFJanssen https://github.com/ChristianFJanssen). I extended this process in GPU (as part of my student job at TU Braunschweig) with JCUDA libraries from http://www.jcuda.de/. 
Now the interactive simualtion is almost 10 times faster with 1GPU of 256 cuda cores compared to 16 threads from CPU. 

Mainly the work was focused on Navierstokes solver only. (Not implemented on other solver models)

Implementation can be found under the following path

/jFlowsim/src/jflowsim/model/numerics/lbm/navierstokes/*.java files
