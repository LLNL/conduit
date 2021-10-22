// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mpi_mesh_flatten.cpp
///
//-----------------------------------------------------------------------------
#include "conduit_blueprint_mpi_mesh_flatten.hpp"

//-----------------------------------------------------------------------------
// std lib includes
//-----------------------------------------------------------------------------
#include <cmath>
#include <iostream>
#include <unordered_set>

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit_log.hpp"
#include "conduit_relay_mpi.hpp"

using conduit::utils::log::quote;

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
ParallelMeshFlattener::ParallelMeshFlattener(MPI_Comm comm)
    : MeshFlattener(), comm(comm)
{
    // Invoke MeshFlattener to set defaults
}

//-----------------------------------------------------------------------------
ParallelMeshFlattener::~ParallelMeshFlattener()
{
    // Nothing special yet
}

//-----------------------------------------------------------------------------
void
ParallelMeshFlattener::flatten_many_domains(const Node &mesh, Node &output) const
{
    const int root = 0;
    const int rank = relay::mpi::rank(comm);

    MeshInfo info;
    try
    {
        collect_mesh_info(mesh, info);
    }
    catch(const conduit::Error &e)
    {
        CONDUIT_INFO("Error caught trying to collect_mesh_info on rank " << rank
            << ", this rank will not contribute to the final table. The error: "
            << e.message());
        info = MeshInfo();
    }

    const std::string topo_name = get_topology(mesh[0]).name();
    std::vector<std::string> fields_to_flatten;

    try
    {
        get_fields_to_flatten(mesh, topo_name, fields_to_flatten);
    }
    catch(const conduit::Error &e)
    {
        CONDUIT_INFO("Error caught trying to get_fields_to_flatten on rank " << rank
            << ", this rank will not contribute to the final table. The error: "
            << e.message());
        info = MeshInfo();
        fields_to_flatten.clear();
    }

    std::vector<std::string> field_associations;
    for(const std::string &field_name : fields_to_flatten)
    {
        const Node *ref_node = get_reference_field(mesh, topo_name, field_name);
        if(ref_node)
        {
            field_associations.push_back(ref_node->child("association").as_string());
        }
        else
        {
            // Should never happen, get_fields_to_flatten should only return fields that
            //  a valid "reference field" exists. So this would have to be a bug.
            CONDUIT_INFO("Could not find valid reference field for field " <<
                quote(field_name) << " - assuming vertex association.");
        }
    }

    // Populate info about this rank
    Node my_info;
    my_info["coord_type"].set(info.coord_type);
    my_info["ndomains"].set(info.ndomains);
    my_info["nverts"].set(info.nverts);
    my_info["nelems"].set(info.nelems);
    my_info.add_child("fields").set(DataType::list());
    my_info.add_child("field_associations").set(DataType::list());
    for(index_t i = 0; i < (index_t)fields_to_flatten.size(); i++)
    {
        my_info["fields"].append().set(fields_to_flatten[i]);
        my_info["field_associations"].append().set(field_associations[i]);
    }

    // DELETE ME
    // Just test that this works with varying field lengths
    if(rank == 3)
    {
        my_info["fields"].append().set("test_field");
        my_info["field_associations"].append().set("element");
    }

    // Q: If the user has passed the field_names option can we assume
    //   that they passed the same field names on all ranks?
    // Need to come up with a list of unqiue field names
    Node all_field_info;
    Node all_info;
    relay::mpi::gather_using_schema(my_info, all_info, root, comm);
    if(rank == root)
    {
        std::cout << "Did this work?\n" << all_info.to_json() << std::endl;

        // Build the list of unqiue field names
        std::unordered_set<std::string> field_set;
        all_field_info["field_names"].set(DataType::list());
        all_field_info["field_associations"].set(DataType::list());
        for(index_t i = 0; i < all_info.number_of_children(); i++)
        {
            const Node &fields = all_info[i]["fields"];
            const Node &field_assocs = all_info[i]["field_associations"];
            for(index_t j = 0; j < fields.number_of_children(); j++)
            {
                const std::string field_name = fields[j].as_string();
                const std::string field_assoc = field_assocs[j].as_string();
                if(field_set.count(field_name) == 0)
                {
                    field_set.insert(field_name);
                    all_field_info["field_names"].append().set(field_name);
                    all_field_info["field_associations"].append().set(field_assoc);
                }
            }
        }
    }
    relay::mpi::broadcast_using_schema(all_field_info, root, comm);

    std::cout << "Rank " << rank << " has received the field info:\n"
        << all_field_info.to_json() << std::endl;
}

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mpi::mesh --
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
// -- end conduit --
//-----------------------------------------------------------------------------
