#---------------------------------------------------------------------------
# MASON location, dev environment, aliases and loading screens
#---------------------------------------------------------------------------
if [ -z $MASON_DEFAULT_LOCATION ]; then
    echo "Using local environment.."
else
    source $MASON_DEFAULT_LOCATION
    alias ms="mason switch"
    alias mt="mason toggle"
    if [ -z $MASON_DEV_ENVIRONMENT ]; then
        echo "MASON_DEV_ENVIRONMENT not set!"
        :
    else
        mason load $MASON_DEV_ENVIRONMENT
    fi
fi
if [ -z "$DATASETS_HOME" ]; then
    DATASETS_HOME="$HOME/Datasets"
fi
if [ -z "$EXPERIMENTS_HOME" ]; then
    EXPERIMENTS_HOME="$HOME/Experiments"
fi


echo "ENVIRONMENT:      $(which python)"
echo "EXPERIMENTS_HOME: $EXPERIMENTS_HOME"
echo "DATASETS_HOME:    $DATASETS_HOME"


SHORT_HOST=$(hostname | cut -d"." -f1)
SHORT_USER=$(echo $USER | cut -d"@" -f1)

# this is for performance reasons: We already use multiprocessing for worker threads
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

# clear;

function wait-for-gpu()
{
    # query pid and type of gpu process
    local device_id="$CUDA_VISIBLE_DEVICES"
    local list_of_pid="$(nvidia-smi -i $device_id | tail -n +16 | head -n -1 | sed 's/\s\s*/ /g' | cut -d' ' -f3)"
    local list_of_ptype="$(nvidia-smi -i $device_id | tail -n +16 | head -n -1 | sed 's/\s\s*/ /g' | cut -d' ' -f4)"

    list_of_pid=( $list_of_pid ) # convert to array
    list_of_ptype=( $list_of_ptype ) # convert to array

    # Remember any process of type C (these are real GPU processes)
    local pid=
    for((i=0;i<${#list_of_pid[@]};i+=1)); do
        if [ "${list_of_ptype[$i]}" == "C" ]; then
            pid="${list_of_pid[$i]}"
        fi
    done

    # Test if pid is free
    if [ "$pid" == "" ]; then
        # echo "Loading..."
        :
    else
        local host=$(hostname | cut -d"." -f1)
        local user="$( ps -o uname= -p "${pid}" )"
        local user="$(echo $user | cut -d"@" -f1)"

        echo "INFO: $host/gpu:$device_id is busy! Waiting for process (pid:$pid, u:$user) ..."

        # this command forces to wait until process $pid is finished..
        while s=`ps -p $pid -o s=` && [[ "$s" && "$s" != 'Z' ]]; do
            sleep 60
        done
        echo "Loading..."
    fi
}

echo