// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_generate_data.hpp
///
//-----------------------------------------------------------------------------
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <algorithm>
#include <memory>

#include <conduit_node.hpp>
#include <conduit_blueprint_mesh_examples.hpp>
#include <conduit_relay_io.hpp>

#ifdef CONDUIT_PARALLEL
#include <conduit_relay_mpi_io_blueprint.hpp>
#include <mpi.h>
#else
#include <conduit_relay_io_blueprint.hpp>
#endif

//#define MAKE_UNIQUE(T) std::make_unique<T>()
#define MAKE_UNIQUE(T) std::unique_ptr<T>(new T())

//-----------------------------------------------------------------------------
/// Generate data for a domain
class DomainGenerator
{
public:
    virtual void generate(int domain[3], conduit::Node &n, conduit::Node &opts) = 0;

    void setDims(const int d[3])
    {
        for(int i = 0; i < 3; i++)
           dims[i] = d[i];
    }
    void setDomains(const int d[3])
    {
        for(int i = 0; i < 3; i++)
           domains[i] = d[i];
    }
    void setMeshType(const std::string &m)
    {
        meshType = m;
    }
protected:
    int dims[3]{0,0,0};
    int domains[3]{1,1,1};
    std::string meshType{};
};

//-----------------------------------------------------------------------------
/// Generate data using tiled data generator.
class TiledDomainGenerator : public DomainGenerator
{
public:
    virtual void generate(int domain[3], conduit::Node &n, conduit::Node &opts) override
    {
        constexpr double side = 20.;

        opts["origin/x"] = domain[0] * dims[0] * side;
        opts["origin/y"] = domain[1] * dims[1] * side;
        opts["origin/z"] = domain[2] * dims[2] * side;

        // Selectively create boundaries based on where the domain is the
        // whole set of domains.
        if(domains[0] * domains[1] * domains[2] == 1)
        {
            opts["boundaries/left"] = 1;
            opts["boundaries/right"] = 1;
            opts["boundaries/bottom"] = 1;
            opts["boundaries/top"] = 1;
            if(dims[0] > 0)
            {
                opts["boundaries/back"] = 1;
                opts["boundaries/front"] = 1;
            }
        }
        else
        {
            opts["boundaries/left"] = ((domain[0] == 0) ? 1 : 0);
            opts["boundaries/right"] = ((domain[0] == domains[0]-1) ? 1 : 0);
            opts["boundaries/bottom"] = ((domain[1] == 0) ? 1 : 0);
            opts["boundaries/top"] = ((domain[1] == domains[1]-1) ? 1 : 0);
            if(dims[0] > 0)
            {
                opts["boundaries/back"] = ((domain[2] == 0) ? 1 : 0);
                opts["boundaries/front"] = ((domain[2] == domains[2]-1) ? 1 : 0);
            }
        }

        conduit::blueprint::mesh::examples::tiled(dims[0], dims[1], dims[2], n, opts);
    }
};

//-----------------------------------------------------------------------------
/// Generate data using braid data generator.
class BraidDomainGenerator : public DomainGenerator
{
public:
    virtual void generate(int /*domain*/[3], conduit::Node &n, conduit::Node &) override
    {
        conduit::blueprint::mesh::examples::braid(meshType, dims[0], dims[1], dims[2], n);

        // TODO: Use domain,domains to adjust coordinates to get them to line up nicely.
    }
};

//-----------------------------------------------------------------------------
void
printUsage(const char *exeName)
{
    std::cout << "Usage: " << exeName << "[-dims x,y,z] [-domains x,y,z] [-tile]\n"
              << "   [-braid] [-output fileroot] [-protocol name] [-meshtype type]\n"
              << "   [-tiledef filename] [-help]\n";
    std::cout << "\n";
    std::cout << "Argument              Description\n";
    std::cout << "===================   ==========================================================\n";
    std::cout << "-dims x,y,z           The number of mesh zones in each dimension. For 2D meshes,\n";
    std::cout << "                      supply 0 for z.\n";
    std::cout << "\n";
    std::cout << "-domains x,y,z        The number of domains in each dimension.\n";
    std::cout << "\n";
    std::cout << "-tile                 Generate a mesh using the tiled data generator.\n";
    std::cout << "\n";
    std::cout << "-braid                Generate a mesh using the braid data generator.\n";
    std::cout << "\n";
    std::cout << "-output fileroot      Specify the root used in filenames that are created.\n";
    std::cout << "\n";
    std::cout << "-protocol name        Specify the protocol used in writing the data. The default\n";
    std::cout << "                      is \"hdf5\".\n";
    std::cout << "\n";
    std::cout << "-meshtype type        The mesh type used when generating data using braid.\n";
    std::cout << "\n";
    std::cout << "-tiledef filename     A file containing a tile definition.\n";
    std::cout << "\n";
    std::cout << "-help                 Print the usage and exit.\n";
}

//-----------------------------------------------------------------------------
int
main(int argc, char *argv[])
{
    int rank = 0;
#ifdef CONDUIT_PARALLEL
    int size = 1;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif

    // Some basic arg parsing.
    int dims[3] = {10, 10, 10};
    int domains[3] = {1, 1, 1};
    conduit::Node n, opts;
    std::unique_ptr<DomainGenerator> g = MAKE_UNIQUE(TiledDomainGenerator);
    std::string meshType("quad"), output("output"), protocol("hdf5");
    for(int i = 1; i < argc; i++)
    {
        if(strcmp(argv[i], "-dims") == 0 && (i+1) < argc)
        {
            int d[3] = {1,1,1};
            if(sscanf(argv[i+1], "%d,%d,%d", &d[0], &d[1], &d[2]) == 3)
            {
                dims[0] = std::max(d[0], 1);
                dims[1] = std::max(d[1], 1);
                dims[2] = std::max(d[2], 1);
            }
            i++;
        }
        else if(strcmp(argv[i], "-domains") == 0 && (i+1) < argc)
        {
            int d[3] = {1,1,1};
            if(sscanf(argv[i+1], "%d,%d,%d", &d[0], &d[1], &d[2]) == 3)
            {
                domains[0] = std::max(d[0], 1);
                domains[1] = std::max(d[1], 1);
                domains[2] = std::max(d[2], 1);
            }
            i++;
        }
        else if(strcmp(argv[i], "-tile") == 0)
        {
            g = MAKE_UNIQUE(TiledDomainGenerator);
        }
        else if(strcmp(argv[i], "-braid") == 0)
        {
            g = MAKE_UNIQUE(BraidDomainGenerator);
        }
        else if(strcmp(argv[i], "-output") == 0 && (i+1) < argc)
        {
            output = argv[i+1];
            i++;
        }
        else if(strcmp(argv[i], "-protocol") == 0 && (i+1) < argc)
        {
            protocol = argv[i+1];
            i++;
        }
        else if(strcmp(argv[i], "-meshtype") == 0 && (i+1) < argc)
        {
            meshType = argv[i+1];
            i++;
        }
        else if(strcmp(argv[i], "-tiledef") == 0 && (i+1) < argc)
        {
            // Load a tile definition file into the options tile node.
            conduit::relay::io::load(argv[i+1], opts["tile"]);
            i++;
        }
        else if(strcmp(argv[i], "-help") == 0 ||
                strcmp(argv[i], "--help") == 0)
        {
            if(rank == 0)
                printUsage(argv[0]);
#ifdef CONDUIT_PARALLEL
            MPI_Finalize();
#endif
            return 0;
        }
    }

    g->setDims(dims);
    g->setDomains(domains);
    g->setMeshType(meshType);

    int ndoms = domains[0] * domains[1] * domains[2];
    if(ndoms == 1 && rank == 0)
    {
        // Single domain.
        int domain[] = {0, 0, 0};
        g->generate(domain, n, opts);
    }
    else
    {
        int domainid = 0;
        for(int k = 0; k < domains[2]; k++)
        {
            for(int j = 0; j < domains[1]; j++)
            {
                for(int i = 0; i < domains[0]; i++, domainid++)
                {
                    int domain[] = {i, j, k};

#ifdef CONDUIT_PARALLEL
                    if(domainid % size == rank)
                    {
#endif
                        // Make the new domain.
                        char domainName[32];
                        sprintf(domainName, "domain_%07d", domainid);
                        conduit::Node &d = n[domainName];
                        g->generate(domain, d, opts);
#ifdef CONDUIT_PARALLEL
                    }
#endif
                }
            }
        }
    }

    // Save the output domains.
#ifdef CONDUIT_PARALLEL
    conduit::relay::mpi::io::blueprint::save_mesh(n, output, protocol, MPI_COMM_WORLD);

    MPI_Finalize();
#else
    conduit::relay::io::save(n, output + "-inspect.yaml", "yaml");
    conduit::relay::io::blueprint::save_mesh(n, output, protocol);
#endif

    return 0;
}
