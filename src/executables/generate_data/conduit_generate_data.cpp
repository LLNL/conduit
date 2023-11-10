// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_generate_data.hpp
///
//-----------------------------------------------------------------------------
#include <cstdlib>
#include <cstdio>
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
    virtual ~DomainGenerator() = default;
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
    void setExtents(const double d[3])
    {
        for(int i = 0; i < 6; i++)
           extents[i] = d[i];
    }
    void setMeshType(const std::string &m)
    {
        meshType = m;
    }
protected:
    int dims[3]{0,0,0};
    int domains[3]{1,1,1};
    double extents[6]{0., 1., 0., 1., 0., 1.};
    std::string meshType{};
};

//-----------------------------------------------------------------------------
/// Generate data using tiled data generator.
class TiledDomainGenerator : public DomainGenerator
{
public:
    virtual void generate(int domain[3], conduit::Node &n, conduit::Node &opts) override
    {
        // Determine the size and location of this domain in the whole.
        double sideX = (extents[1] - extents[0]) / static_cast<double>(domains[0]);
        double sideY = (extents[3] - extents[2]) / static_cast<double>(domains[1]);
        double sideZ = (extents[5] - extents[4]) / static_cast<double>(domains[2]);
        double domainExt[] = {extents[0] + domain[0]     * sideX,
                              extents[0] + (domain[0]+1) * sideX,
                              extents[2] + domain[1]     * sideY,
                              extents[2] + (domain[1]+1) * sideY,
                              extents[4] + domain[2]     * sideZ,
                              extents[4] + (domain[2]+1) * sideZ};
        opts["extents"].set(domainExt, 6);
        opts["domain"].set(domain, 3);
        opts["domains"].set(domains, 3);

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
        conduit::blueprint::mesh::examples::braid(meshType, dims[0] + 1, dims[1] + 1, dims[2] + 1, n);

        // TODO: Use domain,domains to adjust coordinates to get them to line up nicely.
    }
};

//-----------------------------------------------------------------------------
void
printUsage(const char *exeName)
{
    std::cout << "Usage: " << exeName << "[-dims x,y,z] [-domains x,y,z] [-tile]\n"
              << "   [-braid] [-output fileroot] [-protocol name] [-meshtype type]\n"
              << "   [-tiledef filename] [-extents x0,x1,y0,y1[,z0,z1]] [-help]\n";
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
    std::cout << "-extents ext          A list of 4 or 6 comma-separated values indicating extents\n";
    std::cout << "                      as pairs of min,max values for each dimension.\n";
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
    double extents[6] = {0., 1., 0., 1., 0., 1.};
    conduit::Node n, opts;
    std::unique_ptr<DomainGenerator> g = MAKE_UNIQUE(TiledDomainGenerator);
    std::string meshType("hexs"),meshTypeDefault("hexs"), output("output"), protocol("hdf5");
    for(int i = 1; i < argc; i++)
    {
        if(strcmp(argv[i], "-dims") == 0 && (i+1) < argc)
        {
            int d[3] = {1,1,1};
            if(sscanf(argv[i+1], "%d,%d,%d", &d[0], &d[1], &d[2]) == 3)
            {
                dims[0] = std::max(d[0], 1);
                dims[1] = std::max(d[1], 1);
                dims[2] = std::max(d[2], 0); // Allow 0 for 2D

                // If we have not set the mesh type, set it according to dimension.
                if(meshType == meshTypeDefault)
                {
                    meshType = (dims[2] > 0) ? "hexs" : "quads";
                }
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
        else if(strcmp(argv[i], "-extents") == 0 && (i+1) < argc)
        {
            double e[6] = {0., 1., 0., 1., 0., 1.};
            if(sscanf(argv[i + 1], "%lg,%lg,%lg,%lg,%lg,%lg", &e[0], &e[1], &e[2], &e[3], &e[4], &e[5]) == 6)
            {
                memcpy(extents, e, 6 * sizeof(double));
            }
            else if(sscanf(argv[i + 1], "%lg,%lg,%lg,%lg", &e[0], &e[1], &e[2], &e[3]) == 4)
            {
                memcpy(extents, e, 4 * sizeof(double));
            }
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
    g->setExtents(extents);

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
                        snprintf(domainName, sizeof(domainName), "domain_%07d", domainid);
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
    conduit::relay::io::blueprint::save_mesh(n, output, protocol);
#endif

    return 0;
}
