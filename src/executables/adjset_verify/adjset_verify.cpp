// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: adjset_verify.cpp
///
//-----------------------------------------------------------------------------

#include <conduit.hpp>
#include <conduit_relay.hpp>

#include <iostream>
#include <cstring>
#include <numeric>
#include <cstdio>

using std::cout;
using std::endl;

// NOTE: This is a serial prototype.

//---------------------------------------------------------------------------
/**
 @brief Turn a conduit node into a string, stripping any quotes that might be 
        stored around the string.
 @param n The input node.
 @return A string that has had its quotes stripped.
 */
std::string
to_string(const conduit::Node &n)
{
    std::string s(n.to_string());
    if(s.find("\"") == 0)
        s = s.substr(1, s.size() - 2);
    return s;
}

//---------------------------------------------------------------------------
std::string
protocolFromFile(const std::string &filename)
{
    std::string p("hdf5");

    if(filename.find("hdf5") != std::string::npos)
        p = "hdf5";
    else if(filename.find("yaml") != std::string::npos)
        p = "yaml";
    else if(filename.find("json") != std::string::npos)
        p = "json";

    return p;
}

//---------------------------------------------------------------------------
/**
 @brief This class lets us ask for a node that contains a domain and we do not
        need to care whether it comes from various files on disk or from the
        root node.
 */
class Domains
{
public:
    //-----------------------------------------------------------------------
    Domains(const conduit::Node &r, const std::string &path) : root(r),
        rootPath(path), maxDomains(-1), domains()
    {
    }

    //-----------------------------------------------------------------------
    const conduit::Node &getRoot() const
    {
        return root;
    }

    //-----------------------------------------------------------------------
    bool isRootFile() const
    {
        return root.has_child("number_of_files") && root.has_child("file_pattern");
    }

    //-----------------------------------------------------------------------
    const conduit::Node *getDomain(int domainId)
    {
        const conduit::Node *retval = nullptr;
        if(isRootFile())
            retval = getDomainFromRootFile(domainId);
        else
            retval = getDomainFromNode(domainId);

        if(retval == nullptr)
        {
            CONDUIT_ERROR("Domain " << domainId << " could not be retrieved.");
        }

        return retval;
    }

    //-----------------------------------------------------------------------
    int getMaxDomains()
    {
        if(maxDomains == -1)
        {
            if(isRootFile())
            {
                maxDomains = root.fetch_existing("number_of_files").value();
            }
            else
            {
                // Count child nodes that begin with domain and contain a topologies node.
                maxDomains = 0;
                for(conduit::index_t i = 0; i < root.number_of_children(); i++)
                {
                    if(strncmp(root[i].name().c_str(), "domain", 6) == 0 &&
                       root[i].has_child("topologies"))
                    {
                        maxDomains++;
                    }
                }
            }
        }
        return maxDomains;
    }

private:
    //-----------------------------------------------------------------------
    std::string domainFilename(int domainId)
    {
        std::string retval;
        if(domainId < 0 || domainId >= getMaxDomains())
        {
            CONDUIT_ERROR(domainId << " is not a valid domain number.");
        }
        std::string pattern = to_string(root.fetch_existing("file_pattern"));
        char *tmp = new char[pattern.size() + 32];
        sprintf(tmp, pattern.c_str(), domainId);
        retval = tmp;
        delete [] tmp;
        return retval;
    }

    //-----------------------------------------------------------------------
    const conduit::Node *getDomainFromRootFile(int domainId)
    {
        const conduit::Node *retval = nullptr;
        // Check whether we've read the domain before.
        std::map<int, conduit::Node *>::const_iterator it = domains.find(domainId);
        if(it == domains.cend())
        {
            // Get the protocol to use for reading.
            std::string protocol("hdf5");
            if(root.has_path("protocol/name"))
                protocol = to_string(root.fetch_existing("protocol/name"));

            // Load the new domain file.
            auto newNode = new conduit::Node;
            std::string filename(conduit::utils::join_file_path(rootPath, domainFilename(domainId)));
            cout << "Reading " << filename << " as " << protocol << endl;
            conduit::relay::io::load(filename, protocol, *newNode);

            // Store it in the map.
            domains[domainId] = newNode;
            retval = newNode;
        }
        else
        {
            retval = it->second;
        }
        return retval;
    }

    //-----------------------------------------------------------------------
    const conduit::Node *getDomainFromNode(int domainId)
    {
        const conduit::Node *retval = nullptr;

        std::stringstream ss;
        ss << "domain" << domainId;
        std::string domName(ss.str());
        if(root.has_child(domName))
            retval = root.fetch_ptr(domName);
        return retval;
    }

private:
    const conduit::Node &root;
    std::string rootPath;
    int maxDomains;
    std::map<int, conduit::Node *> domains;
};

//---------------------------------------------------------------------------
std::vector<const conduit::Node *>
GetAdjsets(Domains &d)
{
    std::vector<const conduit::Node *> adjsets;
    const conduit::Node *srcNode = nullptr;
    if(d.isRootFile())
    {
        if(d.getRoot().has_path("blueprint_index/mesh/adjsets"))
            srcNode = d.getRoot().fetch_ptr("blueprint_index/mesh/adjsets");
    }
    if(srcNode == nullptr)
    {
        auto d0 = d.getDomain(0);
        if(d0->has_child("adjsets"))
        {
            srcNode = d0->fetch_ptr("adjsets");
        }
    }
    if(srcNode != nullptr)
    {
        for(conduit::index_t i = 0; i < srcNode->number_of_children(); i++)
            adjsets.push_back(srcNode->child_ptr(i));
    }
    return adjsets;
}

//---------------------------------------------------------------------------
bool
AllDomainsHaveAdjsets(Domains &d, std::vector<const conduit::Node *> &adjsets)
{
    int ndoms = d.getMaxDomains();
    bool allDomsHaveAdjsets = true;
    for(int domainId = 0; domainId < ndoms; domainId++)
    {
        auto dom = d.getDomain(domainId);
        const conduit::Node &dom_adjsets = dom->fetch_existing("adjsets");

        for(size_t ai = 0; ai < adjsets.size(); ai++)
        {
            std::string name(adjsets[ai]->name());
            if(!dom_adjsets.has_child(name))
            {
                CONDUIT_ERROR("Domain " << domainId << " does not have adjset \""
                              << name << "\"");
                return false;
            }

            const conduit::Node &adjset = *adjsets[ai];
            const conduit::Node &dom_adjset = dom_adjsets.fetch_existing(name);

            // Compare their attributes.
            std::string a0, a1;
            a0 = to_string(adjset["association"]);
            a1 = to_string(dom_adjset["association"]);
            if(a1 != a0)
            {
                CONDUIT_ERROR("Domain " << domainId << " adjset \"" << name << "\" has association \""
                              << a1 << " but it should be " << a0 << ".");
                return false;
            }
            a0 = to_string(adjset["topology"]);
            a1 = to_string(dom_adjset["topology"]);
            if(a1 != a0)
            {
                CONDUIT_ERROR("Domain " << domainId << " adjset \"" << name << "\" has topology \""
                              << a1 << " but it should be " << a0 << ".");
                return false;
            }
        }
    }

    return allDomsHaveAdjsets;
}

//---------------------------------------------------------------------------
/**
 @brief Look up a certain point in the coordset and return its coordinate.

 @param coordset The coordset node.
 @param value The index of the vertex that we want to get.
 @param[out] coord The output coordinate.

 @return True of the point was found; False otherwise.
 */
bool
PointIdToCoordinate(const conduit::Node &coordset, int value, double coord[3])
{
    const conduit::Node &n_x = coordset.fetch_existing("values/x");
    const conduit::Node &n_y = coordset.fetch_existing("values/y");
    auto x = n_x.as_double_accessor();
    auto y = n_y.as_double_accessor();
    if(value > x.number_of_elements())
    {
        CONDUIT_ERROR("Coordset " << coordset.name() << " does not contain element "
            << value << ". That is out of bounds.");
        return false;
    }

    coord[0] = x[value];
    coord[1] = y[value];

    if(coordset.has_path("values/z"))
    {
        const conduit::Node &n_z = coordset.fetch_existing("values/z");
        auto z = n_z.as_double_accessor();
        coord[2] = z[value];
    }
    return true;
}

//---------------------------------------------------------------------------
/**
 @brief Look up a certain point in the coordset and return its id.

 @param coordset The coordset node.
 @param coord The coordinate we're looking for.
 @param[out] ptid The index of the vertex that we want to get.

 @return True of the point was found; False otherwise.
 */
bool
CoordinateToPointId(const conduit::Node &coordset, const double coord[3], int &ptid)
{
    // TODO: adjust based on a fraction of closest distance between points in the coordset.
    constexpr double EPS = 1.e-6;

    ptid = -1;

    const conduit::Node &n_x = coordset.fetch_existing("values/x");
    const conduit::Node &n_y = coordset.fetch_existing("values/y");
    auto x = n_x.as_double_accessor();
    auto y = n_y.as_double_accessor();
    if(coordset.has_path("values/z"))
    {
        const conduit::Node &n_z = coordset.fetch_existing("values/z");
        auto z = n_z.as_double_accessor();

        // TODO: replace this brute force search with a spatial acceleration structure.

        // For now, we compute the distance between the input coord and all points
        // in the coordset. If we get one that has a close enough distance then we
        // can return.
        auto n = x.number_of_elements();
        for(conduit::index_t i = 0; i < n; i++)
        {
            double dx = coord[0] - x[i];
            double dy = coord[1] - y[i];
            double dz = coord[2] - z[i];
            double sqdist = dx*dx + dy*dy + dz*dz;
            if(sqdist < EPS)
            {
                ptid = static_cast<int>(i);
                return true;
            }
        }
    }
    else
    {
        auto n = x.number_of_elements();
        for(conduit::index_t i = 0; i < n; i++)
        {
            double dx = coord[0] - x[i];
            double dy = coord[1] - y[i];
            double sqdist = dx*dx + dy*dy;
            if(sqdist < EPS)
            {
                ptid = static_cast<int>(i);
                return true;
            }
        }
    }
    return false;
}

//---------------------------------------------------------------------------
/**
 A good test of the adjsets might be the following:

 1. Determine the external faces of all domains
 2. Make a global list of those external points and record which domains use them.
 3. For all points that are used 2 or more times, they better appear in adjsets.
 4. Points not in the global list (2+ times) should NOT appear in the adjset.

 */

//---------------------------------------------------------------------------
/**
 @brief Look through the current domain's adjset and see whether it looks ok
        and that we can locate corresponding points in the connected domains.

 @param doms The Domains object that lets us retrieve domains.
 @param domainId The domain id that we're operating on.
 @param adjsetName The name of the adjset we're operating on.
 @param[out] info The node to which we add any error messages.

 @return True if the adjset is good; False otherwise.
 */
bool
VerifyDomainAdjset(Domains &doms, int domainId, const std::string &adjsetName,
    conduit::Node &info)
{
    bool retval = true;
    auto dom = doms.getDomain(domainId);
    std::string adjsetPath("adjsets/" + adjsetName);
    const conduit::Node &adjset = dom->fetch_existing(adjsetPath);
    std::string association = to_string(adjset.fetch_existing("association"));

    std::string topoName = to_string(adjset.fetch_existing("topology"));

    // Get the coordset name for the adjset's topology.
    std::string coordsetName = to_string(dom->fetch_existing("topologies/"+topoName+"/coordset"));
    const conduit::Node &coordset = dom->fetch_existing("coordsets/"+coordsetName);

    const conduit::Node &groups = adjset.fetch_existing("groups");
    if(association == "vertex")
    {
        for(conduit::index_t i = 0; i < groups.number_of_children(); i++)
        {
            const conduit::Node &group = groups[i];
            const conduit::Node &n_neighbors = group.fetch_existing("neighbors");
            const conduit::Node &n_values = group.fetch_existing("values");

            //cout << group.name() << endl;
            //cout << "neighbors="; n_neighbors.print();
            //cout << "values="; n_values.print();
            //cout << endl;

            conduit::int_accessor neighbors = n_neighbors.as_int_accessor();
            conduit::int_accessor values = n_values.as_int_accessor();
            for(int ni = 0; ni < neighbors.number_of_elements(); ni++)
            {
                // Validate the neighbor number.
                int nbr = neighbors[ni];
                if(nbr < 0 || nbr >= doms.getMaxDomains())
                {
                    std::stringstream ss;
                    ss << "Domain " << domainId << " adjset " << adjsetName
                       << " group " << group.name()
                       << " has an invalid neighbor "
                       << nbr << " at index " << ni;
                    info[adjsetName].append().set(ss.str());

                    retval = false;

                    // We can't continue because the neighbor is invalid.
                    continue;
                }

                // Get the neighbor domain and its coordset.
                auto ndom = doms.getDomain(nbr);
                const conduit::Node &ndom_coordset = ndom->fetch_existing("coordsets/"+coordsetName);

                for(int vi = 0; vi < values.number_of_elements(); vi++)
                {
                    int value = values[vi];

                    // Lookup the point in the local domain to get its coordinates.
                    double coord[3];
                    if(!PointIdToCoordinate(coordset, value, coord))
                    {
                        std::stringstream ss;
                        ss << "Domain " << domainId << " adjset " << adjsetName
                           << " group " << group.name()
                           << " has a vertex id " << value
                           << " that could not be looked up in the local coordset "
                           << coordsetName << ".";

                        conduit::Node &vn = info[group.name()].append();
                        vn["message"].set(ss.str());
                        vn["vertex"] = value;
                        vn["index"] = vi;
                        vn["neighbor"] = nbr;
                        vn["coordinate"] = std::vector<double>(coord, coord + 3);

                        retval = false;
                    }

                    // Try to find the coordinate in the neighbor domain and get its id there.
                    int ndom_ptid = -1;
                    if(!CoordinateToPointId(ndom_coordset, coord, ndom_ptid))
                    {
                        std::stringstream ss;
                        ss << "Domain " << domainId << " adjset " << adjsetName
                           << " group " << group.name()
                           << ": vertex " << value
                           << " (" << coord[0] << ", " << coord[1]
                           << ", " << coord[2] << ") at index " << vi
                           << " could not be located in neighbor domain " << nbr << ".";

                        conduit::Node &vn = info[group.name()].append();
                        vn["message"].set(ss.str());
                        vn["vertex"] = value;
                        vn["index"] = vi;
                        vn["neighbor"] = nbr;
                        vn["coordinate"] = std::vector<double>(coord, coord + 3);

                        retval = false;
                    }
                }
            }
        }
    }
    else
    {
        std::stringstream ss;
        ss << "Unsupported adjset association: " << association << ".";
        info.append().set(ss.str());
        retval = false;
    }
    return retval;
}

//---------------------------------------------------------------------------
/**
 @brief Iterate over all domains and each adjset and make sure that the vertices
        in the adjset are in range and match a vertex in other domains.

 @return True on success; False on failure.
 */
bool
VerifyAdjsets(Domains &doms,
    const std::vector<int> &inputDomainList,
    const std::vector<const conduit::Node *> &adjsets,
    conduit::Node &info)
{
    bool retval = true;
    int maxdoms = doms.getMaxDomains();

    // If the domain list is empty, fill it.
    std::vector<int> domainList(inputDomainList);
    if(domainList.empty())
    {
        domainList.resize(maxdoms);
        std::iota(domainList.begin(), domainList.end(), 0);
    }

    // Iterate over the domain list.
    for(int domainId : domainList)
    {
        for(size_t ai = 0; ai < adjsets.size(); ai++)
        {
            std::string name(adjsets[ai]->name());

            // Make a node to contain messages/errors.
            std::stringstream ss;
            ss << "domain" << domainId << "/" << name;
            std::string path(ss.str());
            conduit::Node &domInfo = info[path];

            cout << "Checking domain " << domainId << " adjset " << name << "... ";
            if(!VerifyDomainAdjset(doms, domainId, name, domInfo))
            {
                cout << "FAIL" << endl;
                retval = false;
            }
            else
            {
                cout << "PASS" << endl;
            }
        }
    }
    return retval;
}

//---------------------------------------------------------------------------
void
writePoints(const conduit::Node &info)
{
    for(conduit::index_t domainId = 0; domainId < info.number_of_children(); domainId++)
    {
        const conduit::Node &dom = info[domainId];
        std::string domainName(dom.name());
        std::string filename(domainName + ".3D");

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
                            fprintf(fp, "x y z neighbor\n");
                        }

                        if(fp != nullptr)
                        {
                            conduit::double_accessor da = err["coordinate"].as_double_accessor();
                            int nbr = err["neighbor"].to_int();
                            fprintf(fp, "%lg %lg %lg %d\n", da[0], da[1], da[2], nbr);
                        }
                    }
                }
            }

            if(fp != nullptr)
            {
                fclose(fp);
                cout << "Wrote " << filename << endl;
            }
        }
    }
}

//---------------------------------------------------------------------------
void
printUsage(const char *program)
{
    cout << "Usage: " << program << " -input filename [-protocol p] [-domain dom]" << endl;
    cout << endl;
    cout << "Argument         Description" << endl;
    cout << "================ ============================================================" << endl;
    cout << "-input filename  Set the input filename." << endl;
    cout << endl;
    cout << "-protocol p      Set the protocol that should be used to read the input file." << endl;
    cout << endl;
    cout << "-domain dom      Add a specific list to the domains the program will check. If" << endl;
    cout << "                 no domains are specified then all domains will be checked." << endl;
}

//---------------------------------------------------------------------------
int
main(int argc, char *argv[])
{
    std::string input, protocol;
    std::vector<int> domainList;

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
        else if(strcmp(argv[i], "-protocol") == 0 && (i+1) < argc)
        {
            protocol = argv[i+1];
            i++;
        }
        else if(strcmp(argv[i], "-domain") == 0 && (i+1) < argc)
        {
            domainList.push_back(atoi(argv[i+1]));
            i++;
        }
    }

    if(input.empty())
    {
        printUsage(argv[0]);
        return -1;
    }
    if(protocol.empty())
        protocol = protocolFromFile(input);

    // Print some info about Conduit.
    //cout << conduit::relay::io::about() << endl;

    int retval = 0;
    try
    {
        conduit::Node root;
        conduit::relay::io::load(input, protocol, root);
        //root.print();

        std::string inputpath, inputfile;
        conduit::utils::rsplit_file_path(input, inputfile, inputpath);
        //cout << "inputpath=" << inputpath << endl;

        Domains doms(root, inputpath);

        // Print the adjset names.
        std::vector<const conduit::Node *> adjsets(GetAdjsets(doms));
        cout << "Adjsets: ";
        for(size_t i = 0; i < adjsets.size(); i++)
        {
            if(i > 0)
                cout << ", ";
            cout << "\"" << adjsets[i]->name() << "\"";
        }
        cout << endl;

        // Make sure all domains have the list of adjsets.
        std::string msg("Check if all domains have compatible adjsets... ");
        if(AllDomainsHaveAdjsets(doms, adjsets))
            cout << msg << "PASS" << endl;
        else
        {
            cout << msg << "FAIL" << endl;
            return -2;
        }

        // Look through the adjsets to see if the points are all good.
        std::string msg2("Check adjsets... ");
        conduit::Node info;
        if(VerifyAdjsets(doms, domainList, adjsets, info))
            cout << msg2 << "PASS" << endl;
        else
        {
            cout << msg2 << "FAIL: The adjsets contain errors." << endl;
            info.print();
            writePoints(info);
            return -3;
        }
    }
    catch(std::exception &err)
    {
        cout << err.what() << endl;
        retval = -3;
    }

    return retval;;
}
