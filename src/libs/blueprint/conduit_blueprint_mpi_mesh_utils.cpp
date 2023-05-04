// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mpi_mesh_util.hpp
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// conduit lib includes
//-----------------------------------------------------------------------------
#include "conduit.hpp"
#include "conduit_blueprint_mpi_mesh_utils.hpp"
#include "conduit_blueprint_mesh.hpp"
#include "conduit_annotations.hpp"
#include "conduit_relay_mpi.hpp"

//#define DEBUG_PRINT
#ifdef DEBUG_PRINT
// NOTE: if DEBUG_PRINT is defined then the library must also depend on conduit_relay
#include "conduit_relay_io.hpp"
#endif

#include <cstring>

//-----------------------------------------------------------------------------
// -- begin conduit --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint --
//-----------------------------------------------------------------------------
namespace blueprint
{

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mpi --
//-----------------------------------------------------------------------------
namespace mpi
{

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mpi::mesh --
//-----------------------------------------------------------------------------
namespace mesh
{

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mpi::mesh::utils --
//-----------------------------------------------------------------------------
namespace utils
{

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mpi::mesh::utils::query --
//-----------------------------------------------------------------------------
namespace query
{

//---------------------------------------------------------------------------
PointQuery::PointQuery(const conduit::Node &mesh, MPI_Comm comm) :
    conduit::blueprint::mesh::utils::query::PointQuery(mesh), m_comm(comm)
{
}

//---------------------------------------------------------------------------
void
PointQuery::Execute(const std::string &coordsetName)
{
    CONDUIT_ANNOTATE_MARK_FUNCTION;

    int rank, size;
    MPI_Comm_rank(m_comm, &rank);
    MPI_Comm_size(m_comm, &size);

    // Figure out which ranks own the domains. Get the domain ids on this rank
    // and sum the number of domains on all ranks.
    std::vector<int> localDomains = DomainIds();
    int ndoms = localDomains.size(), ntotal_doms = 0;
    MPI_Allreduce(&ndoms, &ntotal_doms, 1, MPI_INT, MPI_SUM, m_comm);

    // Make sure each rank knows who owns which domains. This assumes that
    // the domains were set properly and only occur on one rank.
    std::vector<int> local_domain_to_rank(ntotal_doms, 0);
    std::vector<int> domain_to_rank(ntotal_doms, 0);
    for(auto dom : localDomains)
        local_domain_to_rank[dom] = rank;
    MPI_Allreduce(&local_domain_to_rank[0], &domain_to_rank[0], ntotal_doms, MPI_INT, MPI_SUM, m_comm);

    // Put together the queries that we need to run.
    std::vector<int> queries;
    for(auto it = m_domInputs.begin(); it != m_domInputs.end(); it++)
    {
        queries.push_back(rank);              // the rank asking the question
        queries.push_back(it->first);         // the domain
        queries.push_back(it->second.size()); // the number of points.
    }

    // Let all ranks know the sizes of the queries vectors.
    int nq = queries.size();
    std::vector<int> qsize(size, 0);
    MPI_Allgather(&nq, 1, MPI_INT, &qsize[0], 1, MPI_INT, m_comm);

    // Make an offset array and total the sizes.
    std::vector<int> qoffset(size, 0);
    for(int i = 1; i < size; i++)
        qoffset[i] = qoffset[i-1] + qsize[i-1];
    int total_qsize = 0;
    for(int i = 0; i < size; i++)
        total_qsize += qsize[i];

    // Gather the queries to all ranks.
    std::vector<int> allqueries(total_qsize, 0);
    MPI_Allgatherv(&queries[0], nq, MPI_INT,
                   &allqueries[0], &qsize[0], &qoffset[0], MPI_INT,
                   m_comm);

#ifdef DEBUG_PRINT
    // Print out some debugging information to files.
    conduit::Node debug;
    debug["rank"] = rank;
    debug["size"] = size;
    debug["localDomains"].set(localDomains);
    debug["ntotal_doms"] = ntotal_doms;
    debug["domain_to_rank"].set(domain_to_rank);
    debug["queries"].set(queries);
    debug["nq"] = nq;
    debug["qsize"].set(qsize);
    debug["qoffset"].set(qoffset);
    debug["total_qsize"] = total_qsize;
    debug["allqueries"].set(allqueries);

    // Store the points that we'll ask of each domain.
    conduit::Node &qi = debug["inputs"];
    for(auto it = m_domInputs.begin(); it != m_domInputs.end(); it++)
    {
        std::vector<double> qvals(it->second.begin(), it->second.end());
        conduit::Node &qid = qi.append();
        qid["domain"] = it->first;
        qid["input"].set(qvals);
    }

    // Save the input mesh
    debug["mesh"].set_external(m_mesh);

    std::stringstream ss;
    ss << "pointquery." << rank << ".yaml";
    std::string filename(ss.str());
    conduit::relay::io::save(debug, filename, "yaml");
#endif

    // Do a pass upfront where we look for domain ids from the query that
    // do not map to ranks. This would be possible if the adjset references
    // domains that were not actually added to the mesh. This should not
    // happen but it came up in a test case.
    for(size_t i = 0; i < allqueries.size(); i += 3)
    {
        int domain = allqueries[i+1];
        if(domain < 0 || domain >= domain_to_rank.size())
        {
            CONDUIT_ERROR("An adjacency set referenced domain " << domain
                << ", which is not a valid domain in the input mesh.");
        }
    }

    // Now we can start to issue some queries. We do a first pass that sends/recvs
    // all of the query points. Then a second pass does the queries and sends/recvs
    // the results. The results get stored in m_domResults.
    std::map<std::pair<int,int>, conduit::Node *> input_sends, result_sends,
                                                  input_recvs, result_recvs;
    int inputs_tag = 55000000;
    int results_tag = 66000000;
    for(int pass = 0; pass < 2; pass++)
    {
        conduit::relay::mpi::communicate_using_schema C(m_comm);
#ifdef DEBUG_PRINT
        // Turn on logging.
        std::stringstream ss;
        ss << "mpi_pointquery_pass" << pass;
        C.set_logging_root(ss.str());
        C.set_logging(true);
#endif
        for(size_t i = 0; i < allqueries.size(); i += 3)
        {
            int asker  = allqueries[i];
            int domain = allqueries[i+1];
            int npts   = allqueries[i+2];
            int owner = domain_to_rank[domain];

            if(asker == rank)
            {
                if(owner == rank)
                {
                    // This rank already owns the data. We can do the search.
                    // There is no need to communicate.
                    if(pass == 0)
                    {
                        // Query the domain and store the results in r.
                        std::vector<int> &r = m_domResults[domain];
                        r.resize(npts);
                        const conduit::Node *dom = GetDomain(domain);
                        FindPointsInDomain(*dom, coordsetName, Inputs(domain), r);
                    }
                }
                else
                {
                    auto id = std::make_pair(owner, domain);

                    // A different rank owns the data. We need to send the coordinates
                    // we're querying and then post a receive for the result.
                    if(pass == 0)
                    {
                        // Wrap the inputs as a Conduit node and post an isend.
                        const std::vector<double> &inputs = Inputs(domain);
                        input_sends[id] = new conduit::Node;
                        conduit::Node &q = *input_sends[id];
                        q["inputs"].set_external(const_cast<double *>(&inputs[0]), inputs.size());
                        C.add_isend(q, owner, inputs_tag + domain);
                    }

                    if(pass == 1)
                    {
                        // Make a node to receive the results.
                        result_recvs[id] = new conduit::Node;
                        conduit::Node &r = *result_recvs[id];
                        C.add_irecv(r, owner, results_tag + domain);
                    }
                }
            }
            else if(owner == rank)
            {
                // This rank is not asker but it owns the domain and so it has to
                // do the query.
                auto id = std::make_pair(asker, domain);

                if(pass == 0)
                {
                    // Make a node to store the query parameters
                    input_recvs[id] = new conduit::Node;
                    conduit::Node &q = *input_recvs[id];
                    C.add_irecv(q, asker, inputs_tag + domain);
                }

                if(pass == 1)
                {
                    // We have the query parameters in input_recv. Turn it into a vector.
                    conduit::Node &q = *input_recvs[id];
                    auto acc = q["inputs"].as_double_accessor();
                    std::vector<double> input;
                    input.reserve(acc.number_of_elements());
                    for(conduit::index_t i = 0; i < acc.number_of_elements(); i++)
                        input.push_back(acc[i]);

                    // Do the query.
                    std::vector<int> result;
                    const conduit::Node *dom = GetDomain(domain);
                    FindPointsInDomain(*dom, coordsetName, input, result);

                    // Make a node to send the results back to the asker.
                    result_sends[id] = new conduit::Node;
                    conduit::Node &r = *result_sends[id];
                    r["results"].set(result);
                    C.add_isend(r, asker, results_tag + domain);
                }
            }
        }

        // Do the exchanges.
        C.execute();
    }

    // We have query results to convert/store in m_domResults.
    for(auto it = result_recvs.begin(); it != result_recvs.end(); it++)
    {
        int domain = it->first.second;
        const conduit::Node &r = it->second->fetch_existing("results");
        auto acc = r.as_int_accessor();
        std::vector<int> &result = m_domResults[domain];
        result.reserve(acc.number_of_elements());
        for(conduit::index_t i = 0; i < acc.number_of_elements(); i++)
            result.push_back(acc[i]);

#ifdef DEBUG_PRINT
        // Add the results into the debug node. Any points that could not be
        // located will contain -1 (NotFound) for their entry.
        conduit::Node &rn = debug["results"].append();
        rn["domain"] = domain;
        rn["result"].set(result);
#endif
    }

#ifdef DEBUG_PRINT
    // Overwrite the log file with new information.
    conduit::relay::io::save(debug, filename, "yaml");
#endif

    // Clean up maps.
    for(auto it = input_recvs.begin(); it != input_recvs.end(); it++)
        delete it->second;
    for(auto it = input_sends.begin(); it != input_sends.end(); it++)
        delete it->second;
    for(auto it = result_recvs.begin(); it != result_recvs.end(); it++)
        delete it->second;
    for(auto it = result_sends.begin(); it != result_sends.end(); it++)
        delete it->second;
}

//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
MatchQuery::MatchQuery(const conduit::Node &mesh, MPI_Comm comm) :
    conduit::blueprint::mesh::utils::query::MatchQuery(mesh), m_comm(comm)
{
}

//---------------------------------------------------------------------------
void
MatchQuery::Execute()
{
    CONDUIT_ANNOTATE_MARK_FUNCTION;

    int rank, size;
    MPI_Comm_rank(m_comm, &rank);
    MPI_Comm_size(m_comm, &size);

    // Build the query geometries. Store them in the query_topo node.
    CONDUIT_ANNOTATE_MARK_BEGIN("build");
    std::string shape;
    for(auto it = m_query.begin(); it != m_query.end(); it++)
    {
        int dom = it->first.first;
        int query_dom = it->first.second;

        if(shape.empty())
        {
            // We have not determined the shape yet. Do that now so the subset
            // topologies can be built.
            const auto dtopo = GetDomainTopology(dom);
            conduit::blueprint::mesh::utils::ShapeCascade c(*dtopo);
            const auto &s = c.get_shape((c.dim == 0) ? c.dim : (c.dim - 1));
            shape = s.type;
        }

        it->second.builder->Execute(it->second.query_mesh, shape);
    }
    CONDUIT_ANNOTATE_MARK_END("build");

    // If we have a bunch of requests A,B  C,D then we need to get the entity
    // arrays for B,A and D,C. The second int indicates the rank that owns the
    // entity data.

    CONDUIT_ANNOTATE_MARK_BEGIN("setup");
    // Make a set of ints <dom, query_dom, nents> for each map entry. Each
    // rank owns the domains indicated by dom.
    const int ntuple_values = 3;
    std::vector<int> queries;
    queries.reserve(m_query.size() * ntuple_values);
    for(auto it = m_query.begin(); it != m_query.end(); it++)
    {
        queries.push_back(rank);
        queries.push_back(it->first.first);
        queries.push_back(it->first.second);
    }

    // Let all ranks know the sizes of the queries vectors.
    int nq = queries.size();
    std::vector<int> qsize(size, 0);
    MPI_Allgather(&nq, 1, MPI_INT, &qsize[0], 1, MPI_INT, m_comm);

    // Make an offset array and total the sizes.
    std::vector<int> qoffset(size, 0);
    for(int i = 1; i < size; i++)
        qoffset[i] = qoffset[i-1] + qsize[i-1];
    int total_qsize = 0;
    for(int i = 0; i < size; i++)
        total_qsize += qsize[i];

    // Gather the queries to all ranks.
    std::vector<int> allqueries(total_qsize, 0);
    MPI_Allgatherv(&queries[0], nq, MPI_INT,
                   &allqueries[0], &qsize[0], &qoffset[0], MPI_INT,
                   m_comm);

    // Look up a rank that owns a domain.
    auto domain_to_rank = [&ntuple_values](const std::vector<int> &allqueries, int d) -> int
    {
        for(size_t i = 0; i < allqueries.size(); i += ntuple_values)
        {
            int owner = allqueries[i];
            int domain = allqueries[i + 1];
            //int query_domain = allqueries[i + 2];
            if(domain == d)
                return owner;
        }
        return -1;
    };

#ifdef DEBUG_PRINT
    // Print out some debugging information to files.
    conduit::Node debug;
    debug["rank"] = rank;
    debug["size"] = size;
    debug["queries"].set(queries);
    debug["nq"] = nq;
    debug["qsize"].set(qsize);
    debug["qoffset"].set(qoffset);
    debug["total_qsize"] = total_qsize;
    debug["allqueries"].set(allqueries);

    // Store the points that we'll ask of each domain.
    conduit::Node &qi = debug["inputs"];
    for(auto it = m_query.begin(); it != m_query.end(); it++)
    {
        conduit::Node &qid = qi.append();
        qid["domain"] = it->first.first;
        qid["query_domain"] = it->first.second;
        qid["query_topo"] = it->second.query_mesh;
    }

    std::stringstream ss;
    ss << "matchquery." << rank << ".yaml";
    std::string filename(ss.str());
    conduit::relay::io::save(debug, filename, "yaml");
#endif
    CONDUIT_ANNOTATE_MARK_END("setup");

    // Send/recv entity data.
    CONDUIT_ANNOTATE_MARK_BEGIN("communication");
    conduit::relay::mpi::communicate_using_schema C(m_comm);
#ifdef DEBUG_PRINT
    // Turn on logging.
    C.set_logging_root("mpi_matchquery");
    C.set_logging(true);
#endif
    int query_tag = 77000000;    
    for(size_t i = 0; i < allqueries.size(); i += ntuple_values)
    {
        int owner = allqueries[i];
        int domain = allqueries[i + 1];
        int query_domain = allqueries[i + 2];

        auto oppositeKey = std::make_pair(query_domain, domain);

        if(owner == rank)
        {
            // The query was asked by this rank. We need to prepare to
            // receive the opposite query, which is the answer.

            if(m_query.find(oppositeKey) == m_query.end())
            {
                int remote = domain_to_rank(allqueries, query_domain);

                // The domain whose entities we're querying is on a remote rank.
                // We need to receive entities from it.
                auto &q = m_query[oppositeKey];
                conduit::Node &r = q.query_mesh;
                C.add_irecv(r, remote, query_tag + query_domain);
            }
        }
        else
        {
            // We're not the rank that requested the data.
            // See if we own the answer. If so, send. Note the key swap.

            auto it = m_query.find(oppositeKey);
            if(it != m_query.end())
            {
                int remote = domain_to_rank(allqueries, domain);

                // Send the query_topo to the remote rank.
                conduit::Node &s = it->second.query_mesh;
                C.add_isend(s, remote, query_tag + query_domain);
            }
        }
    }

    // Do the communication.
    C.execute();
    CONDUIT_ANNOTATE_MARK_END("communication");

    // Now, m_query should contain query topos for A,B and B,A. For all the
    // domains owned by this rank, make the results.
    CONDUIT_ANNOTATE_MARK_BEGIN("results");
    for(size_t i = 0; i < allqueries.size(); i += ntuple_values)
    {
        int owner = allqueries[i];
        int domain = allqueries[i + 1];
        int query_domain = allqueries[i + 2];

        if(owner == rank)
        {
            auto key = std::make_pair(domain, query_domain);
            auto oppositeKey = std::make_pair(query_domain, domain);

            // Try and get the geometries.
            auto it = m_query.find(key);
            auto oppit = m_query.find(oppositeKey);
            if(it == m_query.end() || oppit == m_query.end())
            {
                CONDUIT_ERROR("MatchQuery is missing the topologies for "
                    << domain << ":" << query_domain);
            }

            // Get both of the topologies.
            conduit::Node &mesh1 = it->second.query_mesh;
            conduit::Node &mesh2 = oppit->second.query_mesh;
            std::string topoKey("topologies/" + topoName);
            conduit::Node &topo1 = mesh1[topoKey];
            conduit::Node &topo2 = mesh2[topoKey];

            // Perform the search and store the results.
            it->second.results = conduit::blueprint::mesh::utils::topology::search(topo2, topo1);

#ifdef DEBUG_PRINT
            conduit::Node &n = debug["queries"].append();
            n["domain_id"] = it->first.first;
            n["query_domain"] = it->first.second;
            n["mesh"].set_external(mesh1);
            n["query_mesh"].set_external(mesh2);
            n["results"].set_external(it->second.results);
#endif
        }
    }
    CONDUIT_ANNOTATE_MARK_END("results");

#ifdef DEBUG_PRINT
    // Overwrite the old file with the recv results.
    conduit::relay::io::save(debug, filename, "yaml");
#endif
}

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mpi::mesh::utils::query --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mpi::mesh::utils::adjset --
//-----------------------------------------------------------------------------

bool
adjset::validate(const Node &doms,
                 const std::string &adjsetName,
                 Node &info,
                 MPI_Comm comm)
{
    auto to_string = [](const conduit::Node &n) -> std::string
    {
        std::string s(n.to_string());
        if(s.find("\"") == 0)
            s = s.substr(1, s.size() - 2);
        return s;
    };

    auto agree = [](bool res, MPI_Comm comm) -> bool
    {
        int size = 1;
        MPI_Comm_size(comm, &size);
        int val = res ? 1 : 0;
        int globalval = 0;
        MPI_Allreduce(&val, &globalval, 1, MPI_INT, MPI_SUM, comm);
        return globalval == size;
    };

    // Make sure that there are multiple domains.
    bool retval = false;
    bool md = conduit::blueprint::mesh::is_multi_domain(doms);
    // Make sure all ranks agree on the answer.
    md = agree(md, comm);
    if(!md)
    {
        info["errors"].append().set("The dataset is not multidomain.");
        return retval;
    }

    // Decide whether all ranks have data.
    auto domains = conduit::blueprint::mesh::domains(doms);

    // We need to figure out the association, topologyName, and coordsetName.
    // Get them from the domains, if available.
    std::string association, topologyName, coordsetName;
    if(!domains.empty())
    {
        const auto dom = domains[0];

        std::string adjsetPath("adjsets/" + adjsetName);
        const conduit::Node &adjset = dom->fetch_existing(adjsetPath);

        association = to_string(adjset.fetch_existing("association"));
        topologyName = to_string(adjset.fetch_existing("topology"));

        const conduit::Node &topo = dom->fetch_existing("topologies/"+topologyName);
        coordsetName = to_string(topo["coordset"]);
    }

    // Some ranks might not have domains from which to figure out association,
    // etc. Make a YAML string with the answers if we have association. Then
    // allreduce on the characters so ranks with no data will get their buffers
    // updated. We use a fixed size buffer.
    constexpr int MAX_BUFFER_SIZE = 1024;
    char *local = new char[MAX_BUFFER_SIZE];
    memset(local, 0, sizeof(char) * MAX_BUFFER_SIZE);
    if(!association.empty())
    {
        std::stringstream ss;
        ss << "association: " << association << std::endl
           << "topology: " << topologyName << std::endl
           << "coordset: " << coordsetName << std::endl;
        std::string msg(ss.str());
        if(msg.size()+1 >= MAX_BUFFER_SIZE)
        {
            delete [] local;
            CONDUIT_ERROR("Message size exceeded max " << MAX_BUFFER_SIZE);
        }
        strncpy(local, msg.c_str(), msg.size());
    }
    char *global = new char[MAX_BUFFER_SIZE];
    memset(global, 0, sizeof(char) * MAX_BUFFER_SIZE);
    MPI_Allreduce(local, global, MAX_BUFFER_SIZE, MPI_CHAR, MPI_MAX, comm);
    global[MAX_BUFFER_SIZE - 1] = '\0';
    conduit::Node s;
    s.parse(global, "yaml");
    delete [] local;
    delete [] global;
    association = s["association"].as_string();
    topologyName = s["topology"].as_string();
    coordsetName = s["coordset"].as_string();

    // Make parallel queries and do the validation.
    conduit::blueprint::mpi::mesh::utils::query::PointQuery PQ(doms, comm);
    conduit::blueprint::mpi::mesh::utils::query::MatchQuery MQ(doms, comm);

    // Do the validation.
    retval = conduit::blueprint::mesh::utils::adjset::validate(doms,
                 adjsetName, association, topologyName, coordsetName,
                 info, PQ, MQ);

    // Make sure all ranks agree on the answer.
    retval = agree(retval, comm);
    return retval;
}

//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mpi::mesh::utils::adjset --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mpi::mesh::utils --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mpi::mesh::utils --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mpi --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------
