#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10  #maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --output=process-%j.out  # %N for node name, %j for jobID
#SBATCH --time=00-03:00      # time (DD-HH:MM)
#SBATCH --mail-user=x2019cwn@stfx.ca # used to send email
#SBATCH --mail-type=ALL

cd /home/jueqi/projects/def-jlevman/U-Net_MRI-Data/
tar -cf all_ADNI_data.tar ADNI/