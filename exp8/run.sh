source /home/spack/spack/share/spack/setup-env.sh
spack load ucx

srun -n 8 ./main $1
# srun -n 8 vtune -collect hotspots -trace-mpi -result-dir task1 -- ./main $1
