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
# 1. Build the cosmos_reason1._cpp module. Most likely need to use srun to schedule an interactive node, and then run the following commands under the interactive node to build the cosmos_reason1._cpp module in the root cosmos-reason1 path.
#    cd cosmos-reason1
#    pip install -e .
# 2. Change the paths in the following ### Needs to Change ### section.
# After the above two steps, now the configuration of the sript is complete.
# We can simply use sbatch cosmos_job_single_node.sh to launch the cosmos slurm jobs on one node.

echo "JOBID $SLURM_JOB_ID"
echo "Using ${NUM_POLICY_NODES} policy nodes and ${NUM_ROLLOUT_NODES} rollout nodes, TOTAL_NODES: ${TOTAL_NODES}"

MOUNTS="/lustre:/lustre/,[[REPO_ROOT_PATH]]:/opt/cosmos-reason1,${HOME}/.cache/huggingface:/root/.cache/huggingface"

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
    cd /opt/cosmos-reason1
    ./tools/launch_controller.sh --port ${CONTROLLER_PORT} --config [[CONFIG_PATH]]
    ' \
    &
pid_controller=$!

export LOCAL_NODE_LIST=${POLICY_NODES}
# Start policy nodes
srun \
    --overlap \
    --nodes="${NUM_POLICY_NODES}" \
    --nodelist="${LOCAL_NODE_LIST}" \
    --container-image [[COSMOS_CONTAINER]] \
    --container-mounts "${MOUNTS}" \
    --no-container-mount-home \
    --export=ALL \
    -o ${OUTDIR}/%j/policy/%t.out \
    -e ${OUTDIR}/%j/policy/%t.err \
    bash -c \
    '
    cd /opt/cosmos-reason1
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
    cd /opt/cosmos-reason1
    python ./tools/slurm/cosmos_reason1_slurm_launch.py --type rollout
    ' \
    &
pid_rollout=$!

echo "Waiting for policy and rollout jobs to end. If fails, will cancel at ${SLURM_JOB_ID}"

# Monitor both
while true; do
    kill -0 $pid_policy 2>/dev/null
    pol_alive=$?

    kill -0 $pid_rollout 2>/dev/null
    roll_alive=$?

    kill -0 $pid_controller 2>/dev/null
    crl_alive=$?

    if [ $pol_alive -ne 0 ] && [ $roll_alive -ne 0 ] && [ $crl_alive -ne 0 ]; then
        # Both are no longer running — check their exit codes
        wait $pid_policy
        exit_code_policy=$?

        wait $pid_rollout
        exit_code_rollout=$?

        wait $pid_controller
        exit_code_controller=$?

        if [ $exit_code_policy -ne 0 ] || [ $exit_code_rollout -ne 0 ] || [ $exit_code_controller -ne 0 ]; then
            echo "One or both jobs failed"
            scancel $SLURM_JOB_ID
            exit 1
        else
            echo "All jobs succeeded"
            exit 0
        fi
    fi

    # If one finished, check its status
    if [ $pol_alive -ne 0 ]; then
        # Policy ended — check if it failed
        wait $pid_policy
        ec=$?
        if [ $ec -ne 0 ]; then
            echo "Policy failed. Killing rollout."
            kill $pid_rollout 2>/dev/null || true
            kill $pid_controller 2>/dev/null || true
            scancel $SLURM_JOB_ID
            exit $ec
        fi
    fi

    if [ $roll_alive -ne 0 ]; then
        # Rollout ended — check if it failed
        wait $pid_rollout
        ec=$?
        if [ $ec -ne 0 ]; then
            echo "Rollout failed. Killing policy."
            kill $pid_policy 2>/dev/null || true
            kill $pid_controller 2>/dev/null || true
            scancel $SLURM_JOB_ID
            exit $ec
        fi
    fi

    if [ $crl_alive -ne 0 ]; then
        # Controller ended — check if it failed
        wait $pid_controller
        ec=$?
        if [ $ec -ne 0 ]; then
            echo "Controller failed. Killing policy and rollout."
            kill $pid_policy 2>/dev/null || true
            kill $pid_rollout 2>/dev/null || true
            scancel $SLURM_JOB_ID
            exit $ec
        fi
    fi

    sleep 1
done
