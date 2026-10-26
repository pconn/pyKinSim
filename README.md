# pyKinSim

Code to implement analyses for the manuscript "Improving approximate Bayesian computation for close-kin mark-recapture:
bias reduction and tensor parallelism"


BATCH SIZE DETERMINATION
An example analysis for determining appropriate batch sizes for individual-based kinship simulations via GPU computing with PyTorch

\\inst\\benchmark\_pyKinSim\_batch.py


BENCHMARKING CODE EXECUTION TIMES
Examines computing time for PyTorch simulations via GPU parallelism vs. fishSim and CKMRpop employing CPU parallelism.
This study is run using the script 

\\inst\\benchmark\_all.R

SIMULATION STUDY
Re-analyzes simulation study 2 of Conn et al. 2025 (Environmetrics 36:e70049) using GPU computing combined with alternative summary
statistics based on kinship proportions instead of absolute values.  This study is run with 

\\inst\\sim\_study2.R



