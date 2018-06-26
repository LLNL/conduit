#!/bin/bash

# Installation location of DataSpaces
DATASPACES_DIR=$HOME/Development/IL/AICR/thirdparty_static/dataspaces/1.6.2/i386-apple-darwin16_clang

# Number of processors
PRODUCER_NP=2
CONSUMER_NP=2
TOTAL_NP=4

# Number of time steps
NTS=10

function start_dataspaces
{
    # Make the dataspaces.conf file.
    echo "#Config file for dataspaces\n" > dataspaces.conf
    echo "ndim=3\n" >> dataspaces.conf
    echo "dims = 200,200,200\n" >> dataspaces.conf
    echo "max_versions=${NTS}\n" >> dataspaces.conf

    # Start DataSpaces server.
    echo "Starting DataSpaces server..."
    mpirun -np $PRODUCER_NP $DATASPACES_DIR/bin/dataspaces_server -s $NP -c $TOTAL_NP & #>& server.log

    # Wait for the server to start up.
    while [ ! -f conf ]; do
        echo "Waiting for servers to start up."
        sleep 2s
    done
    # Read the server conf file.
    while read line; do
        echo "${line}"
        export set "${line}"
    done  < conf
    echo "DataSpaces has started."
}

# Clean up
rm -f server.log srv.lck dataspaces.conf conf output*.json

# Start DataSpaces server (a function so it is easier to comment out)
#start_dataspaces

# Start the staging program that produces and consumes the data.
mpirun -np $TOTAL_NP ./conduit_staging --split $PRODUCER_NP:$CONSUMER_NP --nts $NTS
