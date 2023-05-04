// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: adjset_verify.cpp
///
//-----------------------------------------------------------------------------

#include <conduit.hpp>
#include <conduit_blueprint.hpp>
#include <conduit_relay.hpp>
#include <conduit_blueprint_mesh_topology_metadata.hpp>
#include <conduit_blueprint_mesh_utils.hpp>

#include <iostream>
#include <cstring>
#include <numeric>
#include <cstdio>

// NOTE: This is a serial prototype.

//---------------------------------------------------------------------------
std::vector<const conduit::Node *>
GetAdjsets(const conduit::Node &doms)
{
    std::vector<const conduit::Node *> adjsets;
    auto domains = conduit::blueprint::mesh::domains(doms);
    for(const auto &dom : domains)
    {
        if(dom->has_child("adjsets"))
        {
            const auto srcNode = dom->fetch_ptr("adjsets");
            for(conduit::index_t i = 0; i < srcNode->number_of_children(); i++)
                adjsets.push_back(srcNode->child_ptr(i));
            break;
        }
    }
    return adjsets;
}

//---------------------------------------------------------------------------
/**
 @brief If the error information included coordinates then we write those to
        a 3D points file for VisIt so we can look at where the invalid points
        occur.

 @param info A node that contains error messages about the adjset.
 */
void
writePoints(const conduit::Node &info)
{
    for(conduit::index_t domainId = 0; domainId < info.number_of_children(); domainId++)
    {
        const conduit::Node &dom = info[domainId];
        std::string domainName(dom.name());

        for(conduit::index_t adjsetId = 0; adjsetId < dom.number_of_children(); adjsetId++)
        {
            const conduit::Node &adjset = dom[adjsetId];

            std::string filename(domainName + "_" + adjset.name() + ".3D");
            FILE *fp = nullptr;

            for(conduit::index_t groupId = 0; groupId < adjset.number_of_children(); groupId++)
            {
                const conduit::Node &group = adjset[groupId];
                for(conduit::index_t errId = 0; errId < group.number_of_children(); errId++)
                {
                    const conduit::Node &err = group[errId];
                    if(err.has_child("coordinate") && err.has_child("neighbor"))
                    {
                        if(fp == nullptr)
                        {
                            fp = fopen(filename.c_str(), "wt");
                            fprintf(fp, "x y z vertex\n");
                        }

                        if(fp != nullptr)
                        {
                            conduit::double_accessor da = err["coordinate"].as_double_accessor();
                            int v = err["vertex"].to_int();
                            //int v = err["neighbor"].to_int();
                            fprintf(fp, "%lg %lg %lg %d\n", da[0], da[1], da[2], v);
                        }
                    }
                }
            }

            if(fp != nullptr)
            {
                fclose(fp);
                std::cout << "Wrote " << filename << std::endl;
            }
        }
    }
}

//---------------------------------------------------------------------------
void
printUsage(const char *program)
{
    std::cout << "Usage: " << program << " -input filename [-help]" << std::endl;
    std::cout << std::endl;
    std::cout << "Argument         Description" << std::endl;
    std::cout << "================ ============================================================" << std::endl;
    std::cout << "-input filename  Set the input filename." << std::endl;
    std::cout << std::endl;
    std::cout << "-help            Print the usage and exit." << std::endl;
    std::cout << std::endl;
}

//---------------------------------------------------------------------------
int
main(int argc, char *argv[])
{
    std::string input;
    for(int i = 1; i < argc; i++)
    {
        if(strcmp(argv[i], "-help") == 0 || strcmp(argv[i], "-h") == 0)
        {
            printUsage(argv[0]);
            return -1;
        }
        else if(strcmp(argv[i], "-input") == 0 && (i+1) < argc)
        {
            input = argv[i+1];
            i++;
        }
    }

    if(input.empty())
    {
        printUsage(argv[0]);
        return -1;
    }

    // Print some info about Conduit.
    //std::cout << conduit::relay::io::about() << std::endl;

    int retval = 0;
    try
    {
        conduit::Node root;
        conduit::relay::io::blueprint::load_mesh(input, root);
        //root.print();

        // Get adjsets.
        std::vector<const conduit::Node *> adjsets(GetAdjsets(root));

        // Look through the adjsets to see if the points are all good.
        std::string msg2("Check adjset ");
        bool err = false;
        for(size_t i = 0; i < adjsets.size(); i++)
        {
            std::string adjsetName(adjsets[i]->name());
            conduit::Node info;
            bool res = conduit::blueprint::mesh::utils::adjset::validate(root, adjsetName, info);
            if(res)
            {
                std::cout << msg2 << adjsetName << "... PASS" << std::endl;
            }
            else
            {
                std::cout << msg2 << adjsetName << "... FAIL: The adjsets contain errors." << std::endl;
                info.print();
                writePoints(info);
                err = true;
            }
        }
        if(err)
            return -2;
    }
    catch(std::exception &err)
    {
        std::cout << err.what() << std::endl;
        retval = -3;
    }

    return retval;
}
