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
bool
AllDomainsHaveAdjsets(const conduit::Node &doms, std::vector<const conduit::Node *> &adjsets)
{
    auto domains = conduit::blueprint::mesh::domains(doms);

    int ndoms = domains.size();
    bool allDomsHaveAdjsets = true;
    for(const auto dom : domains)
    {
        int domainId = 0;
        if(dom->has_path("state/domain_id"))
            domainId = dom->fetch_existing("state/domain_id").to_int();

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

 @param doms The node that contains all of the domains.
 @param domainId The domain id that we're operating on.
 @param adjsetName The name of the adjset we're operating on.
 @param[out] info The node to which we add any error messages.

 @return True if the adjset is good; False otherwise.
 */
bool
VerifyDomainAdjset(const conduit::Node &doms, int domainId,
    const std::string &adjsetName, conduit::Node &info)
{
    auto domains = conduit::blueprint::mesh::domains(doms);
    auto getDomainById = [](const std::vector<const conduit::Node *> &domains, int domainId) -> const conduit::Node *
    {
        for(auto dom : domains)
        {
            if(dom->has_path("state/domain_id"))
            {
                if(dom->fetch_existing("state/domain_id").to_int() == domainId)
                    return dom;
            }
        }
        return nullptr;
    };

    bool retval = true;
    auto dom = getDomainById(domains, domainId);
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
                if(nbr < 0 || nbr >= domains.size())
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
                auto ndom = getDomainById(domains, nbr);
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
/*!
 * @brief Accepts a node that has multiple domains and determines whether the
 *        specified adjset is set up properly by using queries.
 */
bool
adjset_verify(const conduit::Node &doms,
              const std::string &adjsetName,
              conduit::Node &info)
{
    using index_t = conduit::index_t;
    using TopologyMetadata = conduit::blueprint::mesh::utils::TopologyMetadata;

    bool retval = false;
    auto domains = conduit::blueprint::mesh::domains(doms);

cout << "adjset_verify: adjsetName=" << adjsetName << endl;

    // Determine whether the adjset exists in the first domain.
    std::string adjsetPath("adjsets/" + adjsetName);
    const auto dom0 = domains[0];
    if(!dom0->has_path(adjsetPath))
        return retval;

cout << "adjset_verify: domains.size=" << domains.size() << endl;

    // Use the first domain in the list to determine the adjset association.
    const conduit::Node &adjset = dom0->fetch_existing(adjsetPath);
    std::string association = to_string(adjset.fetch_existing("association"));
    std::string topologyName = to_string(adjset.fetch_existing("topology"));
cout << "adjset_verify: association=" << association << endl;
cout << "adjset_verify: topologyName=" << topologyName << endl;
cout << "adjset_verify: topologies={";
const conduit::Node &topos = dom0->fetch_existing("topologies");
for(int i = 0; i < topos.number_of_children(); i++)
    cout << topos[i].name() << ", ";
cout << "}" << endl;
adjset.print();

    if(association == "vertex")
    {
        conduit::blueprint::mesh::utils::query::PointQuery Q(doms);

        // Iterate over the domains so we can add their adjset points to the
        // point query.
        std::string coordsetName;
        std::vector<std::tuple<index_t, index_t, index_t, index_t, std::string, std::vector<double>>> query_guide;
        for(auto dom : domains)
        {
            auto domainId = conduit::blueprint::mesh::utils::find_domain_id(*dom);

            // Get the domain's topo and coordset.
            const conduit::Node &topo = dom->fetch_existing("topologies/"+topologyName);
            coordsetName = topo["coordset"].as_string();
            const conduit::Node &coordset = dom->fetch_existing("coordsets/"+coordsetName);

            // Get the domain's adjset and groups.
            const conduit::Node &adjset = dom->fetch_existing(adjsetPath);
            const conduit::Node &adjset_groups = adjset.fetch_existing("groups");

            // Iterate over this domain's adjset to help build up the point query.
            for(const std::string &group_name : adjset_groups.child_names())
            {
                const conduit::Node &src_group = adjset_groups[group_name];
                conduit::index_t_accessor src_neighbors = src_group["neighbors"].value();
                conduit::index_t_accessor src_values = src_group["values"].value();

                // Neighbors
                for(index_t ni = 0; ni < src_neighbors.dtype().number_of_elements(); ni++)
                {
                    int nbr = src_neighbors[ni];
                    // Point ids
                    for(index_t pi = 0; pi < src_values.dtype().number_of_elements(); pi++)
                    {
                        // Look up the point in the local coordset to get the coordinate.
                        int ptid = src_values[pi];
                        auto pt = conduit::blueprint::mesh::utils::coordset::_explicit::coords(coordset, ptid);
                        double pt3[3];
                        pt3[0] = pt[0];
                        pt3[1] = (pt.size() > 1) ? pt[1] : 0.;
                        pt3[2] = (pt.size() > 2) ? pt[2] : 0.;

                        // Ask domain nbr if they have point pt3
                        auto idx = Q.Add(nbr, pt3);
                        query_guide.emplace_back(domainId, ptid, nbr, idx, group_name, pt);
                    }
                }
            }
        }

        // Execut the query.
        Q.Execute(coordsetName);

        // Iterate over the query results to flag any problems.
        retval = true;
        for(const auto &obj : query_guide)
        {
            index_t domain_id = std::get<0>(obj);
            index_t ptid = std::get<1>(obj);
            index_t nbr = std::get<2>(obj);
            index_t idx = std::get<3>(obj);
            const std::string &group_name = std::get<4>(obj);
            const std::vector<double> &coord = std::get<5>(obj);

            const auto &res = Q.Results(domain_id);
            if(res[idx] == conduit::blueprint::mesh::utils::query::PointQuery::NotFound)
            {
                retval = false;
                std::stringstream dss;
                dss << "domain_" << domain_id;
                std::string dname(dss.str());

                conduit::Node &vn = info[dname][adjsetName][group_name].append();

                std::stringstream ss;
                ss << "Domain " << domain_id << " adjset " << adjsetName
                   << " group " << group_name
                   << ": vertex " << ptid
                   << " (" << coord[0] << ", " << coord[1]
                   << ", " << coord[2] << ") at index " << ptid
                   << " could not be located in neighbor domain "
                   << nbr << ".";

                vn["message"].set(ss.str());
                vn["vertex"] = ptid;
                vn["neighbor"] = nbr;
                vn["coordinate"] = coord;
            }
        }
    }
    else if(association == "element")
    {
        // Make topology metadata for each domain so we can access the lower
        // level topology (external surfaces/edges). Make a new extdoms node
        // that contains these as domains.
        std::map<int, TopologyMetadata *> topo_mds;
        conduit::Node extdoms;
        for(size_t i = 0; i < domains.size(); i++)
        {
            const conduit::Node *dom = domains[i];
            auto domain_id = conduit::blueprint::mesh::utils::find_domain_id(*dom);

            // Get the domain's topo and coordset of interest.
            const conduit::Node &topo = dom->fetch_existing("topologies/"+topologyName);
            std::string coordsetName = topo["coordset"].as_string();
            const conduit::Node &coordset = dom->fetch_existing("coordsets/"+coordsetName);

            // Get the topology dimension.
            size_t dim = conduit::blueprint::mesh::utils::topology::dims(topo);
            size_t lower_dim = dim - 1;

            // Produce the lower topology via cascade.
            std::vector<std::pair<size_t,size_t> > desired_maps;
            desired_maps.push_back(std::make_pair(dim, lower_dim));
            desired_maps.push_back(std::make_pair(lower_dim, dim));
            topo_mds[domain_id] = new TopologyMetadata(topo, coordset, lower_dim, desired_maps);

            // Reference the lower topo in extdoms.
            std::stringstream ss;
            ss << "domain_" << domain_id;
            std::string domKey(ss.str());
            extdoms[domKey]["state/domain_id"] = domain_id;
            const auto &lower_topo = topo_mds[domain_id]->get_topology(lower_dim);
            extdoms[domKey]["topologies/" + topologyName].set_external(lower_topo);
        }

        // Make a MatchQuerty that will examine the extdoms domains.
        conduit::blueprint::mesh::utils::query::MatchQuery Q(extdoms);
        Q.SelectTopology(topologyName);

        std::vector<std::tuple<int, int, int, conduit::uint64>> query_guide;
        for(auto dom : domains)
        {
            auto domain_id = conduit::blueprint::mesh::utils::find_domain_id(*dom);

            // Get the domain's adjset and groups.
            const conduit::Node &adjset = dom->fetch_existing(adjsetPath);
            const conduit::Node &adjset_groups = adjset.fetch_existing("groups");

            // Get the domain's topo and coordset.
            const conduit::Node &topo = dom->fetch_existing("topologies/"+topologyName);
            std::string coordsetName = topo["coordset"].as_string();
            const conduit::Node &coordset = dom->fetch_existing("coordsets/"+coordsetName);

            // Get the topology dimension.
            size_t dim = conduit::blueprint::mesh::utils::topology::dims(topo);
            size_t lower_dim = dim - 1;

            // Iterate over the adjset data to build up neighbors to points map.
            std::map<index_t, std::set<index_t>> neighbor_pidxs_map;
            for(const std::string &group_name : adjset_groups.child_names())
            {
                const conduit::Node &group = adjset_groups[group_name];
                conduit::index_t_accessor neighbors = group["neighbors"].value();
                conduit::index_t_accessor values    = group["values"].value();

                for(index_t ni = 0; ni < neighbors.dtype().number_of_elements(); ni++)
                {
                    std::set<index_t> &neighbor_pidxs = neighbor_pidxs_map[neighbors[ni]];
                    for(index_t pi = 0; pi < values.dtype().number_of_elements(); pi++)
                    {
                        neighbor_pidxs.insert(values[pi]);
                    }
                }
            }

            // We need the external faces (or edges) for this domain. We then
            // iterate over them to see whether all of an entity's points are
            // in the adjset. If so, it is a candidate that we'll check for
            // validity (i.e. Does the entity also exist in the neighbor domain?).
            const auto topo_md = topo_mds[domain_id];
            const auto &lower_topo = topo_md->get_topology(lower_dim);
            index_t lower_topo_nelem = conduit::blueprint::mesh::utils::topology::length(lower_topo);
            for(index_t ei = 0; ei < lower_topo_nelem; ei++)
            {
                // if we are dealing with anything but points
                // we don't want to include duplicated entities 
                // (that means they are internal to the domain)
                std::vector<index_t> entity_pidxs = conduit::blueprint::mesh::utils::topology::unstructured::points(lower_topo, ei);
                const auto d2s_sizes = topo_md->get_global_association(ei, lower_dim, dim);
                if(d2s_sizes.size() < 2 || entity_pidxs.size() == 1)
                {
                    for(const auto &neighbor_pair : neighbor_pidxs_map)
                    {
                        const index_t &ni = neighbor_pair.first;
                        const std::set<index_t> &neighbor_pidxs = neighbor_pair.second;

                        // check if the new element has all of its points
                        // contained inside of the adjset
   
                        bool entity_in_neighbor = true;
                        for(index_t pi = 0; pi < (index_t)entity_pidxs.size() && entity_in_neighbor; pi++)
                        {
                            entity_in_neighbor &= neighbor_pidxs.find(entity_pidxs[pi]) != neighbor_pidxs.end();
                        }

                        // if the element is fully in the adjset, add to query.
                        if(entity_in_neighbor)
                        {
                            // Add the entity to the query for consideration.
                            conduit::uint64 qid = Q.Add(domain_id, ni, entity_pidxs);

                            // Add the candidate entity to the membership query, which
                            // will help resolve things across domains.
                            query_guide.push_back(std::make_tuple(domain_id, ni, ei, qid));
                        }
                    }
                }
            }
        }

        Q.Execute();

        // Iterate over the query results to flag any problems.
        retval = true;
        for(const auto &obj : query_guide)
        {
            int domain_id = std::get<0>(obj);
            int nbr = std::get<1>(obj);
            int ei = std::get<2>(obj);
            conduit::uint64 eid = std::get<3>(obj);

            if(!Q.Exists(domain_id, nbr, eid))
            {
                retval = false;
                std::stringstream dss;
                dss << "domain_" << domain_id;
                std::string dname(dss.str());

                conduit::Node &vn = info[dname][adjsetName].append();

                std::stringstream ss;
                ss << "Domain " << domain_id << " adjset " << adjsetName
                   << ": element " << ei << " could not be located in neighbor domain "
                   << nbr << ".";

                vn["message"].set(ss.str());
                vn["element"] = ei;
                vn["neighbor"] = nbr;
            }
        }

        // Clean up
        extdoms.reset();
        for(auto it = topo_mds.begin(); it != topo_mds.end(); it++)
            delete it->second;
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
VerifyAdjsets(const conduit::Node &doms,
    const std::vector<const conduit::Node *> &adjsets,
    conduit::Node &info)
{
    bool retval = true;
    info.reset();
    // Iterate over the adjsets.
    for(size_t ai = 0; ai < adjsets.size(); ai++)
    {
        std::string name(adjsets[ai]->name());
        adjset_verify(doms, name, info);
    }
    return info.number_of_children() == 0;
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
                            int nbr = err["vertex"].to_int();
                            //int nbr = err["neighbor"].to_int();
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
    }

    if(input.empty())
    {
        printUsage(argv[0]);
        return -1;
    }

    // Print some info about Conduit.
    //cout << conduit::relay::io::about() << endl;

    int retval = 0;
    try
    {
        conduit::Node root;
        conduit::relay::io::blueprint::load_mesh(input, root);
        //root.print();

        // Print the adjset names.
        std::vector<const conduit::Node *> adjsets(GetAdjsets(root));
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
        if(AllDomainsHaveAdjsets(root, adjsets))
            cout << msg << "PASS" << endl;
        else
        {
            cout << msg << "FAIL" << endl;
            return -2;
        }

        // Look through the adjsets to see if the points are all good.
        std::string msg2("Check adjsets... ");
        conduit::Node info;
        if(VerifyAdjsets(root, adjsets, info))
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
