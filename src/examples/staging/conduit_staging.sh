#!/bin/bash

# Installation location of DataSpaces
#DATASPACES_DIR=$HOME/Development/IL/AICR/thirdparty_static/dataspaces/1.6.2/i386-apple-darwin16_clang
DATASPACES_DIR=$HOME/Development/AICR/thirdparty_static/dataspaces/1.6.2

#LAUNCHCMD="mpirun -np"
LAUNCHCMD="srun -n"

# Number of processors
PRODUCER_NP=2
CONSUMER_NP=2
TOTAL_NP=4

# Number of time steps
NTS=10

# Whether to split the conduit_staging program instance into producer/consumer parts
SPLIT="on"
#SPLIT="off"

function start_dataspaces
{
    # Make the dataspaces.conf file.
    echo "#Config file for dataspaces" > dataspaces.conf
    echo "ndim = 3" >> dataspaces.conf
    echo "dims = 200,200,200" >> dataspaces.conf
    echo "hash_version = 2" >> dataspaces.conf
    echo "lock_type = 2" >> dataspaces.conf
    echo "max_versions = 1" >> dataspaces.conf

    # Start DataSpaces server.
    echo "Starting DataSpaces server..."
    echo "$LAUNCHCMD $PRODUCER_NP $DATASPACES_DIR/bin/dataspaces_server -s $PRODUCER_NP -c $TOTAL_NP"
    $LAUNCHCMD $PRODUCER_NP $DATASPACES_DIR/bin/dataspaces_server -s $PRODUCER_NP -c $TOTAL_NP &

    # Wait for the server to start up.
    while [ ! -f conf ]; do
        echo "Waiting for servers to start up."
        sleep 2s
    done
    # Read the server conf file.
    while read line; do
        echo "${line}"
        export ${line}
    done  < conf

    echo "Environment:"
    env | grep P2TNID
    env | grep P2TPID
    echo "DataSpaces has started."
    ps -A | grep dataspaces
    echo "****************************************************"
}

# Clean up
rm -f server.log conf srv.lck dataspaces.conf conf output*.json

# Start DataSpaces server (a function so it is easier to comment out)
start_dataspaces

if test "$SPLIT" = "on" ; then
    # Start the staging program that produces and consumes the data.
    echo "$LAUNCHCMD $TOTAL_NP ./conduit_staging --split $PRODUCER_NP:$CONSUMER_NP --nts $NTS"
    $LAUNCHCMD $TOTAL_NP ./conduit_staging --split $PRODUCER_NP:$CONSUMER_NP --nts $NTS
else
    # Start as separate launches
    echo "$LAUNCHCMD $PRODUCER_NP ./conduit_staging --producer --nts $NTS"
    $LAUNCHCMD $PRODUCER_NP ./conduit_staging --producer --nts $NTS &

    #sleep 2s
    echo "$LAUNCHCMD $CONSUMER_NP ./conduit_staging --consumer --nts $NTS"
    $LAUNCHCMD $CONSUMER_NP ./conduit_staging --consumer --nts $NTS
fi
