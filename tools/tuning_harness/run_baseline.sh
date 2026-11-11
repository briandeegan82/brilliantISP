#!/bin/bash
# Launch the shipped-config baseline with single-threaded native libs per worker.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1
cd ~/isp_tuning/harness
rm -rf ~/isp_tuning/runs/r00v11_baseline_shipped
exec ~/raw-dataset-converter/.venv/bin/python run_eval.py --isp-root ~/raw-dataset-converter/brilliantISP \
  --svs-config config/SVS_cam.yml --ad-config config/AD_cam_2.yml --name r00v11_baseline_shipped --scenes all --workers 12
