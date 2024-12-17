#!/bin/bash
#SBATCH -N 1
#SBATCH -p lrz-hgx-h100-92x4
#SBATCH --gres=gpu:1
#SBATCH --job-name=vmacoSy
#SBATCH -t 1-00:00:00
#SBATCH --mail-user=afshinbigboy@gmail.com
#SBATCH --mail-type=END,FAIL,BEGIN
#SBATCH --container-image="/dss/dssfs04/lwp-dss-0002/pn36fu/pn36fu-dss-0000/env/pytorch-env.sqsh"
#SBATCH --container-mounts="/dss/dssfs04/lwp-dss-0002/pn36fu/pn36fu-dss-0000:/cabinet"


# Define the section to execute
SECTION=$1
if [ -z "$SECTION" ]; then
  echo "Error: No section specified."
  echo "Valid sections are: SYNAPSE"
  exit 1
fi
# Define the fold variable, defaulting to 0 if not provided
fold=${2:-0}

echo "SECTION: $SECTION"
nvidia-smi



# cd 
# Load the environment variables
# ./env_v2.sh

base_path="${HOME}/deeplearning/vmaco"
run_path="${base_path}/train_synapse.py"


case "$SECTION" in
  SYNAPSE)
    echo "Running SYNAPSE-2D section..."
    cd ${base_path}
    pip install -r requirements.txt
    python ${run_path}
    # python ${find_best_configuration_path} 700 -c 3d_fullres -f 0 --disable_ensembling -tr lhunetSynapseTrainer
    ;;
    *)
    echo "Invalid section specified: $SECTION"
    echo "Valid sections are: SYNAPSE"
    exit 1
    ;;
esac
