# AutoDock-Vina-CUDA

AutoDock Vina is a widely used open-source docking engines. During the process of docking, it uses a simple scoring function and rapid gradient-optimization conformational search strategy.  Based on AutoDock-Vina-v1.2.5, using nvidia gpu and cuda SIMT to accelerate the gradient-optimization conformational search process.

## Accelerating Method

1. docking a ligand with multiple blocks and descreasing the monte-carlo steps

2. using SIMT to unroll loop for updating Hessian matrix H and direction vector p

3. using SIMT to unroll loop for line search strategy, then using parallel reduction method to find min step size a

4. unroll loop for sum energy of ligand-ligand atom pairs, then using modulo scheduling mothed to descreasing the number of iterations

5. using cuda cooperative groups to tile block, achieving a multi-level parallelism for outer line search loop, inner summing energy loop, inner setting transformation matrix loop, and inner derivation loop

## Enviroment

- CUDA version: 12.0
- Boost version: 1.77.0
- AutoDock-Vina lib version: 1.2.5

## Usage

1. cd ${AUTODOCK_VINA_GPU_HOME}/build/linux/release

2. make or make clean && make -j32

3. ./vina --config ${INPUT_CONFIG_FILE_PATH}

    the config file format is like this:
```
    receptor = ${RECEPTOR_PDBQT_FILE_PATH}
    ligand = ${LIGAND_PDBQT_FILE_PATH}
    center_x = 
    center_y =
    center_z = 
    size_x =
    size_y =
    size_z = 
```

