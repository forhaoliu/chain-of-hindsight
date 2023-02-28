function tpu {
    trap "trap - SIGINT SIGTERM; return 1;" SIGINT SIGTERM

    # =============== TPU Project Specific Definitions ===============
    export PROJECT_NAME=${PROJECT_NAME:-"N/A"}
    export PROJECT_HOME=${PROJECT_HOME:-"N/A"}
    echo "---------------------------------"
    echo "Project name: $PROJECT_NAME"
    echo "Project home: $PROJECT_HOME"
    echo "---------------------------------"

    if [ "$1" = "YOURS" ]; then
        tpu_project='YOURS'
        tpu_zone='YOURS'
    else
        echo "Invalid syntax!"
        trap - SIGINT SIGTERM
        return 1
    fi
    # =============== End of TPU Project Specific Definitions ===============


    if [ "$2" = "list" ]; then
        gcloud alpha compute tpus tpu-vm list --zone $tpu_zone --project $tpu_project
    elif [ "$2" = "describe" ]; then
        gcloud alpha compute tpus tpu-vm describe $3 --zone $tpu_zone --project $tpu_project
    elif [ "$2" = "ips" ]; then
        _tpu_ips $tpu_zone $tpu_project $3
    elif [ "$2" = "delete" ]; then
        echo "${@:3}"
        for tpu in "${@:3}"; do
            echo -n "Are you sure (y/n)? "
            read REPLY
            if [[ $REPLY =~ ^[Yy]$ ]]
            then
                echo y|gcloud alpha compute tpus tpu-vm delete "$tpu" --zone $tpu_zone --project $tpu_project &
            fi
        done
    elif [ "$2" = "create" ]; then
        _tpu_create $tpu_zone $tpu_project $3 $4
    elif [ "$2" = "retry_create" ]; then
        _tpu_retry_create $tpu_zone $tpu_project $3 $4
    elif [ "$2" = "create_pre" ]; then
        _tpu_create_pre $tpu_zone $tpu_project $3 $4
    elif [ "$2" = "retry_create_pre" ]; then
        _tpu_retry_create_pre $tpu_zone $tpu_project $3 $4
    elif [ "$2" = "cs" ]; then
        _tpu_create $tpu_zone $tpu_project $3 $4
        sleep 90s
        _tpu_setup $tpu_zone $tpu_project $4
    elif [ "$2" = "check" ]; then
        _tpu_check $tpu_zone $tpu_project $3
    elif [ "$2" = "setup" ]; then
        _tpu_setup $tpu_zone $tpu_project $3
    elif [ "$2" = "copy" ]; then
        _tpu_copy $tpu_zone $tpu_project $3
    elif [ "$2" = "stop" ]; then
        _tpu_stop $tpu_zone $tpu_project $3
    elif [ "$2" = "launch" ]; then
        _tpu_launch $tpu_zone $tpu_project $3 $4
    elif [ "$2" = "cl" ]; then
        _tpu_copy $tpu_zone $tpu_project $3
        _tpu_launch $tpu_zone $tpu_project $3 $4
    elif [ "$2" = "rcl" ]; then
        _tpu_reboot $tpu_zone $tpu_project $3
        sleep 180s
        _tpu_copy $tpu_zone $tpu_project $3
        _tpu_launch $tpu_zone $tpu_project $3 $4
    elif [ "$2" = "maintain" ]; then
        _tpu_maintain $tpu_zone $tpu_project $3
    elif [ "$2" = "ssh" ]; then
        _tpu_ssh $tpu_zone $tpu_project $3 "$4"
    elif [ "$2" = "reboot" ]; then
        _tpu_reboot $tpu_zone $tpu_project $3
    elif [ "$2" = "rm" ]; then
        _tpu_rm $tpu_zone $tpu_project $3
    else
        echo "Invalid syntax!"
        trap - SIGINT SIGTERM
        return 1
    fi
    trap - SIGINT SIGTERM
}


function _tpu_ips {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3
    gcloud alpha compute tpus tpu-vm describe $tpu_name --zone $tpu_zone --project $tpu_project | grep -oP 'externalIp: \K(.+)$'
}

function _tpu_create {
    tpu_zone=$1
    tpu_project=$2
    tpu_cores=$3
    tpu_name=$4
    software_version='tpu-vm-base'
    gcloud alpha compute tpus tpu-vm create \
        $tpu_name \
        --accelerator-type="v3-$tpu_cores" \
        --version $software_version \
        --zone $tpu_zone \
        --project $tpu_project
}

function _tpu_create_v4 {
    tpu_zone=$1
    tpu_project=$2
    tpu_cores=$3
    tpu_name=$4
    software_version='tpu-vm-base'
    gcloud alpha compute tpus tpu-vm create \
        $tpu_name \
        --accelerator-type="v4-$tpu_cores" \
        --version $software_version \
        --zone $tpu_zone \
        --project $tpu_project
}

function _tpu_retry_create {
    while true; do
        _tpu_create "$@"
        sleep 30s
    done
}

function _tpu_create_pre {
    tpu_zone=$1
    tpu_project=$2
    tpu_cores=$3
    tpu_name=$4
    software_version='tpu-vm-base'
    gcloud alpha compute tpus tpu-vm create \
        $tpu_name \
        --accelerator-type="v3-$tpu_cores" \
        --version $software_version \
        --zone $tpu_zone \
        --project $tpu_project \
        --preemptible
}

function _tpu_retry_create_pre {
    while true; do
        _tpu_create_pre "$@"
        sleep 30s
    done
}

function _tpu_setup {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3

    tpu_ips=($(echo "$(_tpu_ips $tpu_zone $tpu_project $tpu_name)"))
    for host in $tpu_ips[@]; do
        scp $PROJECT_HOME/$PROJECT_NAME/scripts/tpu_vm_setup.sh $host:~/
        ssh $host '~/tpu_vm_setup.sh' &
    done
    wait &> /dev/null

    for host in $tpu_ips[@]; do
        scp $PROJECT_HOME/$PROJECT_NAME/scripts/tpu_vm_setup.sh $host:~/
        wait
        ssh $host '~/tpu_vm_setup.sh' &
    done
    wait &> /dev/null
}

function _tpu_check {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3

    tpu_ips=($(echo "$(_tpu_ips $tpu_zone $tpu_project $tpu_name)"))
    for host in $tpu_ips[@]; do
        echo "============== Checking host: $host =============="
        ssh $host 'tmux capture-pane -pt launch'
        echo "============== End of host: $host =============="
        echo
        echo
    done
}

function _tpu_copy {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3

    tpu_ips=($(echo "$(_tpu_ips $tpu_zone $tpu_project $tpu_name)"))
    for host in $tpu_ips[@]; do
        rsync -avPI --exclude=logs --exclude=__pycache__ --exclude=.git --exclude=local $PROJECT_HOME/$PROJECT_NAME $host:~/ &
    done
    wait &> /dev/null
    sleep 1s

    for host in $tpu_ips[@]; do
        rsync -avPI --exclude=logs --exclude=__pycache__ --exclude=.git --exclude=local $PROJECT_HOME/$PROJECT_NAME $host:~/ &
    done
    wait &> /dev/null
    sleep 1s
}

function _tpu_stop {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3

    tpu_ips=($(echo "$(_tpu_ips $tpu_zone $tpu_project $tpu_name)"))
    for host in $tpu_ips[@]; do
        ssh $host 'tmux kill-session -t launch ; pkill -9 python' &
    done
    wait &> /dev/null
}

function _tpu_launch {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3
    command=$4

    if [ -z "$command" ]; then
        echo "Invalid syntax!"
        return 1
    fi

    tpu_ips=($(echo "$(_tpu_ips $tpu_zone $tpu_project $tpu_name)"))
    for host in $tpu_ips[@]; do
        ssh $host "tmux new -d -s launch ~/$PROJECT_NAME/jobs/$command" &
    done
    wait &> /dev/null
}

function _tpu_maintain {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3

    gcloud alpha compute tpus tpu-vm simulate-maintenance-event $tpu_name \
        --project $tpu_project \
        --zone=$tpu_zone \
        --workers=all
}

function _tpu_ssh {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3
    command="$4"

    if [ -z "$command" ]; then
        echo "Invalid syntax!"
        return 1
    fi

    tpu_ips=($(echo "$(_tpu_ips $tpu_zone $tpu_project $tpu_name)"))
    for host in $tpu_ips[@]; do
        ssh $host "$command" &
    done
    wait &> /dev/null
}

function _tpu_reboot {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3

    tpu_ips=($(echo "$(_tpu_ips $tpu_zone $tpu_project $tpu_name)"))
    for host in $tpu_ips[@]; do
        ssh $host 'sudo reboot' &
    done
    wait &> /dev/null
}


function _tpu_rm {
    tpu_zone=$1
    tpu_project=$2
    tpu_name=$3

    tpu_ips=($(echo "$(_tpu_ips $tpu_zone $tpu_project $tpu_name)"))
    for host in $tpu_ips[@]; do
        ssh $host 'rm -rf ~/*' &
    done
    wait &> /dev/null
}


function export_function() {
  if [[ -n "${ZSH_VERSION}" ]]; then
    for f in "$@"; do
      zle -N $f
    done
  else
    export -f "$@"
  fi
}

export_function tpu _tpu_ips _tpu_create _tpu_create_pre _tpu_retry_create _tpu_retry_create_pre _tpu_setup _tpu_check _tpu_copy _tpu_stop _tpu_launch _tpu_maintain _tpu_ssh _tpu_reboot _tpu_rm
