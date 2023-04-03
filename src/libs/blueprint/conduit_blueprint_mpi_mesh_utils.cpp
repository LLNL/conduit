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
#include "conduit_annotations.hpp"
#include "conduit_relay_mpi.hpp"

//#define DEBUG_PRINT
#ifdef DEBUG_PRINT
// NOTE: if DEBUG_PRINT is defined then the library must also depend on conduit_relay
#include "conduit_relay_io.hpp"
#endif

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

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mpi::mesh::utils::query --
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
