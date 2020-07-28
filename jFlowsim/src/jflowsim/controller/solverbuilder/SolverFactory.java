package jflowsim.controller.solverbuilder;

import jflowsim.model.numerics.acm.AcmSolver;
import jflowsim.model.numerics.lbm.freesurface.LBMFreeSurfaceSolver;
import jflowsim.model.numerics.lbm.navierstokes.LBMNavierStokesSolver;
import jflowsim.model.numerics.lbm.temperature.LBMTemperatureSolver;
import java.util.ArrayList;
import java.util.TreeMap;
import jflowsim.model.numerics.lbm.shallowwater.LBMShallowWaterSolver;
import jflowsim.model.numerics.lbm.testcases.TestCase;
import jflowsim.model.numerics.lbm.testcases.navierstokes.ChannelFlowTestCase;
import jflowsim.model.numerics.lbm.testcases.navierstokes.CouetteFlowTestCase;
import jflowsim.model.numerics.lbm.testcases.navierstokes.DrivenCavityTestCase;
import jflowsim.model.numerics.lbm.testcases.navierstokes.PoiseuilleTestCase;

public class SolverFactory {

    private TreeMap<String, SolverBuilder> builderSet = new TreeMap<String, SolverBuilder>();
    private static ArrayList<String> solverTypeName = new ArrayList<String>();
    // protected TreeMap<String, TestCase> solverTypeName = new TreeMap<String, TestCase>();
    private static SolverFactory instance;

    private SolverFactory() {
        builderSet.put(LBMNavierStokesSolver.class.getSimpleName(), new LBMNavierStokesBuilder());
        //builderSet.put(LBMFreeSurfaceSolver.class.getSimpleName(), new LBMFreeSurfaceBuilder());
        builderSet.put(LBMTemperatureSolver.class.getSimpleName(), new LBMTemperatureBuilder());
        builderSet.put(LBMShallowWaterSolver.class.getSimpleName(), new LBMShallowWaterBuilder());
        //builderSet.put(AcmSolver.class.getSimpleName(), new AcmBuilder());
       /*solverTypeName.put("CPU", new PoiseuilleTestCase());
       solverTypeName.put("GPU", new PoiseuilleTestCase());
       /*solverTypeName.put("CPU", new ChannelFlowTestCase());
       solverTypeName.put("GPU", new ChannelFlowTestCase());
       
       solverTypeName.put("CPU", new DrivenCavityTestCase());
       solverTypeName.put("GPU", new DrivenCavityTestCase());
       solverTypeName.put("CPU", new CouetteFlowTestCase());
       solverTypeName.put("GPU", new CouetteFlowTestCase());*/
       
       solverTypeName.add("CPU");
       solverTypeName.add("GPU");
    }

    public static SolverFactory getInstance() {
        if (instance == null) {
            instance = new SolverFactory();
        }
        return instance;
    }

    public SolverBuilder getBuilder(String name) {
        return builderSet.get(name);
    }

    public ArrayList<String> getKeySet() {
        return new ArrayList<String>(builderSet.keySet());
    }
    
    public ArrayList<String> getSolverTyp() {
        return (solverTypeName);
    }
}
