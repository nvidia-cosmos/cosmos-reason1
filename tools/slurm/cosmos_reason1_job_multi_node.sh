#!/bin/bash
#SBATCH -A [[SLURM_ACCOUNT]] 
#SBATCH -J [[SLURM_JOB_NAME]]
#SBATCH -t 04:00:00 
#SBATCH --nodes=[[TOTAL_NODES]]
#SBATCH --mem=0 
#SBATCH --gres=gpu:8
#SBATCH --dependency=singleton
#SBATCH -p [[SLURM_PARTITION]]
#SBATCH --output=[[OUTPUT_ROOT_PATH]]/%j/x.out
#SBATCH --error=[[OUTPUT_ROOT_PATH]]/%j/x.err

# Prerequisite of using this slurm script
# 1. Build the cosmos_reason1._cpp module. Most likely need to use srun to schedule an interactive node, and then run the following commands under the interactive node to build the cosmos_reason1._cpp module in the root ref-rl path.
#    cd ref-rl
#    pip install -e .
# 2. Change the paths in the following ### Needs to Change ### section.
# After the above two steps, now the configuration of the sript is complete.
# We can simply use sbatch cosmos_job_single_node.sh to launch the cosmos slurm jobs on one node.

echo "JOBID $SLURM_JOB_ID"
echo "Using ${NUM_POLICY_NODES} policy nodes and ${NUM_ROLLOUT_NODES} rollout nodes, TOTAL_NODES: ${TOTAL_NODES}"

MOUNTS="/lustre:/lustre/,[[REPO_ROOT_PATH]]:/opt/ref-rl,${HOME}/.cache/huggingface:/root/.cache/huggingface"

export OUTDIR="[[OUTPUT_ROOT_PATH]]/${SLURM_JOB_NAME}"
mkdir -p ${OUTDIR}
mkdir -p ${OUTDIR}/${SLURM_JOB_ID}
mkdir -p ${OUTDIR}/${SLURM_JOB_ID}/controller
mkdir -p ${OUTDIR}/${SLURM_JOB_ID}/policy
mkdir -p ${OUTDIR}/${SLURM_JOB_ID}/rollout

export CONTROLLER_PORT=8082
export NODELIST=$(scontrol show hostname $SLURM_JOB_NODELIST)
echo "NODELIST: $NODELIST"

# Use the first policy node for the controller
export POLICY_NODES=$(echo $NODELIST | cut -d' ' -f1-$((NUM_POLICY_NODES)))
export CONTROLLER_NODE=$(echo $POLICY_NODES | cut -d' ' -f1)
export COSMOS_CONTROLLER_HOST="${CONTROLLER_NODE}:${CONTROLLER_PORT}"

# Get rollout nodes
export ROLLOUT_NODES=$(echo $NODELIST | cut -d' ' -f$((NUM_POLICY_NODES+1))-$((TOTAL_NODES)))

# Start controller on first policy node
srun \
    --overlap \
    --nodes=1 \
    --nodelist=${CONTROLLER_NODE} \
    --container-image [[COSMOS_CONTAINER]] \
    --container-mounts ${MOUNTS} \
    --no-container-mount-home \
    --export=ALL \
    -o ${OUTDIR}/%j/controller/%t.out \
    -e ${OUTDIR}/%j/controller/%t.err \
    bash -c \
    '
    # Start the controller
    export COSMOS_LOG_LEVEL=DEBUG
    cd /opt/ref-rl
    ./tools/launch_controller.sh --port ${CONTROLLER_PORT} --config [[CONFIG_PATH]] --log ${OUTDIR}/${SLURM_JOB_ID}/redis_server.log
    ' \
    &

export LOCAL_NODE_LIST=${POLICY_NODES}
# Start policy nodes
srun \
    --overlap \
    --nodes="${NUM_POLICY_NODES}" \
    --nodelist="${LOCAL_NODE_LIST}" \
    --container-image [[COSMOS_CONTAINER]] \
    --no-container-mount-home \
    --container-mounts "${MOUNTS}" \
    --no-container-mount-home \
    --export=ALL \
    -o ${OUTDIR}/%j/policy/%t.out \
    -e ${OUTDIR}/%j/policy/%t.err \
    bash -c \
    '
    cd /opt/ref-rl
    python ./tools/slurm/cosmos_reason1_slurm_launch.py --type policy
    ' \
    &
pid_policy=$!

export LOCAL_NODE_LIST=${ROLLOUT_NODES}
# Start rollout nodes
srun \
    --nodes="${NUM_ROLLOUT_NODES}" \
    --nodelist="${LOCAL_NODE_LIST}" \
    --container-image [[COSMOS_CONTAINER]] \
    --container-mounts "${MOUNTS}" \
    --no-container-mount-home \
    --export=ALL \
    -o ${OUTDIR}/%j/rollout/%t.out \
    -e ${OUTDIR}/%j/rollout/%t.err \
    bash -c \
    '
    cd /opt/ref-rl
    python ./tools/slurm/cosmos_reason1_slurm_launch.py --type rollout
    ' \
    &
pid_rollout=$!

echo "Waiting for policy and rollout jobs to end. If fails, will cancel at ${SLURM_JOB_ID}"
wait $pid_policy $pid_rollout
if [ $? -ne 0 ]; then
    scancel $SLURM_JOB_ID
    exit 1
else
    echo "All jobs succeeded"
fi
