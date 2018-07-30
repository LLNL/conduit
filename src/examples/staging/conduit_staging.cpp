//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2018, Lawrence Livermore National Security, LLC.
// 
// Produced at the Lawrence Livermore National Laboratory
// 
// LLNL-CODE-666778
// 
// All rights reserved.
// 
// This file is part of Conduit. 
// 
// For details, see: http://software.llnl.gov/conduit/.
// 
// Please also read conduit/LICENSE
// 
// Redistribution and use in source and binary forms, with or without 
// modification, are permitted provided that the following conditions are met:
// 
// * Redistributions of source code must retain the above copyright notice, 
//   this list of conditions and the disclaimer below.
// 
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
// 
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, 
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
// POSSIBILITY OF SUCH DAMAGE.
// 
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

//-----------------------------------------------------------------------------
///
/// file: conduit_staging.cpp
///
//-----------------------------------------------------------------------------
#include <conduit.hpp>
#include <conduit_relay_mpi_io.hpp>
#include <conduit_utils.hpp>
#include <iostream>
#include <cstdlib>
#include <cstring>

#ifdef CONDUIT_RELAY_IO_ADIOS_ENABLED
#include <conduit_relay_mpi_io_adios.hpp>
#endif

#ifndef _WIN32
#include <unistd.h>
#endif
#include <mpi.h>

using conduit::Node;
using std::cout;
using std::endl;

//-----------------------------------------------------------------------------
bool streaming_transport(const std::string &transport)
{
    return (transport == "DATASPACES" ||
            transport == "DIMES" || 
            transport == "FLEXPATH" ||
            transport == "ICEE");
}

//-----------------------------------------------------------------------------
std::string write_verb(const std::string &transport)
{
    return streaming_transport(transport) ? "staging" : "writing";
}

//-----------------------------------------------------------------------------
int 
producer(const Node &config, int nts, MPI_Comm comm)
{
    const char *prefix = "producer: ";
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    std::string path(config["path"].as_string());
    std::string selected_options(config["selected_options"].as_string());
    const Node &options = config[selected_options];

    if(rank == 0)
    {
        cout << prefix << "Options: " << options.to_json() << endl;
        cout << prefix << "Starting." << endl;
    }
    // Remove the file if it exists.
    conduit::utils::remove_file(path);

    // Write multiple time steps to the same "file".
    for(int ts = 0; ts < nts; ++ts)
    {
        Node out;
        int idx = ts*100 + rank*10;
        out["a"] = idx + 1;
        out["b"] = idx + 2;
        out["c/d"] = idx + 3;
        out["c/e"] = idx + 4;
        out["f"] = 3.14159f * float(ts);
        if(rank == 0)
            cout << prefix << "Before add_step" << endl;
        // Add a new time step to the output file.
        conduit::relay::mpi::io::add_step(out, path, options, comm);

        // Show some progress
        if(rank == 0)
        {
            cout << prefix
                 << write_verb(options["write/transport"].as_string())
                 << " time step " << ts << endl;
        }
    }

    return 0;
}

//-----------------------------------------------------------------------------
int
consumer(const Node &config, int nts, MPI_Comm comm)
{
    const char *prefix = "consumer: ";
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if(rank == 0)
        cout << prefix << "Starting." << endl;

    char outpath[1024];
    std::string path(config["path"].as_string());
    std::string protocol(config["protocol"].as_string());
    std::string format(config["outpath"].as_string());
    std::string selected_options(config["selected_options"].as_string());
    const Node &options = config[selected_options];

    // Read from the producer and save out.
    for(int ts = 0; ts < nts; ++ts)
    {
        // TODO: query the number of domains, distribute among ranks,
        // read each domain. For now, assume producer and consumer have 
        // same number of ranks.
        int domain = rank;

        if(rank == 0)
            cout << prefix << "reading time step " << ts << endl;

        // Read in a domain of the current time step.
        Node in;

        if(streaming_transport(options["write/transport"].as_string()))
        {
            // Read "current" time step from the stream.
            conduit::relay::mpi::io::load(path, protocol, options, in, comm);
        }
        else
        {
            // Read a specific time step from the file.
            std::ostringstream tspath;
            tspath << path << ":" << ts << ":" << domain;
            conduit::relay::mpi::io::load(tspath.str(), protocol, options, in, comm);
        }

        // Make an output filename and save.
        sprintf(outpath, format.c_str(), ts, domain);
        conduit::relay::mpi::io::save(in, outpath, MPI_COMM_WORLD);

        if(rank == 0)
            cout << prefix << "writing time step " << ts << endl;
    }

    return 0;
}

//-----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int retval = 0;
    std::string configfile("conduit_staging.json");
    int mode = 0;
    int nts = 10;
    int p = 1;
    int c = 1;
    bool split = false;
    for(int i = 1; i < argc; ++i)
    {
        if(strcmp(argv[i], "--config") == 0)
        {
            configfile = std::string(argv[i+1]);
            ++i;
        }
        else if(strcmp(argv[i], "--nts") == 0)
        {
            int ivalue = atoi(argv[i+1]);
            if(ivalue > 0)
                nts = ivalue;
            ++i;
        }
        else if(strcmp(argv[i], "--split") == 0)
        {
            int p0 = 1, c0 = 1;
            bool bs = true;
            if(sscanf(argv[i+1], "%d:%d", &p0, &c0) == 2)
            {
                if((p0 + c0) == size &&
                   p0 > 0 &&
                   c0 > 0
                   && p0 == c0 // The consumer is not coded for M:N right now.
                   )
                {
                    p = p0;
                    c = c0;
                    split = true;
                    bs = false;
                }
            }
            if(bs && rank == 0)
            {
                cout << "Bad --split " << argv[i+1] << endl;
                mode = 3;
                break;
            }
            ++i;
        }
        else if(strcmp(argv[i], "--help") == 0 ||
                strcmp(argv[i], "-h") == 0)
        {
            mode = 2;
        }
        else
        {
            if(rank == 0)
            {
                for(int j = 0; j < argc; j++)
                    cout << argv[j] << endl;
            }
            mode = 3;
            retval = -1;
            break;
        }
    }

    if(mode >= 2)
    {
        if(rank == 0)
        {
            if(mode > 2)
            {
                cout << "ARGS:" << endl;
                for(int j = 0; j < argc; j++)
                    cout << "\t" << argv[j] << endl;
                cout << endl;
            }

            cout << "Usage: " << argv[0] << " [--config file] [--nts number] "
                "[--split p:c] [--help]" << endl;
            cout << endl;
            cout << "--config file  The name of the config file." << endl;
            cout << "--nts number   The number of time steps." << endl;
            cout << "--split p:c    The ratio of producers to consumers." << endl;
            cout << "--help         Print help and exit." << endl;
        }
    }
    else
    {
        conduit::relay::mpi::io::initialize(MPI_COMM_WORLD);

#ifdef CONDUIT_RELAY_IO_ADIOS_ENABLED
        Node opts;
        conduit::relay::mpi::io::adios_options(opts, MPI_COMM_WORLD);
        if(rank == 0)
            cout << "Default ADIOS options = " << opts.to_json() << endl;
#endif
        try
        {
            // Read the config file.
            Node config;
            conduit::relay::mpi::io::load(configfile, config, MPI_COMM_WORLD);
            if(rank == 0)
            {
                cout << "config = " << config.to_json() << endl;
            }

            // Split (or dup) the communicator
            MPI_Comm comm;
            if(split)
                MPI_Comm_split(MPI_COMM_WORLD, rank < p, 0, &comm);
            else
            {
                p = size / 2;
                MPI_Comm_dup(MPI_COMM_WORLD, &comm);
            }

            if(rank < p)
            {
                retval = producer(config, nts, comm);
            }
            else
            {
#ifndef _WIN32
                // If we're using BP files then delay a little until the 
                // producer is done. Streaming blocks until data are 
                // available.
                sleep(10);
#endif
                retval = consumer(config, nts, comm);
            }

            MPI_Comm_free(&comm);
        }
        catch(...)
        {
            cout << "Failed to read config file " << configfile << endl;
            retval = -2;
        }

        conduit::relay::mpi::io::finalize(MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return retval;
}
