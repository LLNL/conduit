//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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
/// file: conduit_relay_io_blueprint.cpp
///
//-----------------------------------------------------------------------------

#include "conduit_relay_io.hpp"
#include "conduit_relay_io_handle.hpp"

#include "conduit_blueprint.hpp"

#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
    #include "conduit_relay_mpi.hpp"
    #include "conduit_relay_mpi_io_blueprint.hpp"
#else
    #include "conduit_relay_io_blueprint.hpp"
#endif


#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
// Define an argument macro that adds the communicator argument.
#define CONDUIT_RELAY_COMMUNICATOR_ARG(ARG) ,ARG
#else
// Define an argument macro that does not add the communicator argument.
#define CONDUIT_RELAY_COMMUNICATOR_ARG(ARG) 
#endif

// std includes
#include <limits>
#include <set>

//-----------------------------------------------------------------------------
// standard lib includes
//-----------------------------------------------------------------------------
#include <iostream>


//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- begin conduit::relay --
//-----------------------------------------------------------------------------
namespace relay
{

#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
//-----------------------------------------------------------------------------
// -- begin conduit::relay::mpi --
//-----------------------------------------------------------------------------
namespace mpi
{
#endif


//-----------------------------------------------------------------------------
// -- begin conduit::relay::io
//-----------------------------------------------------------------------------
namespace io
{
    
//-----------------------------------------------------------------------------
// -- begin conduit::relay::io::<mpi>::blueprint
//-----------------------------------------------------------------------------
namespace blueprint
{

//-----------------------------------------------------------------------------
// -- begin conduit::relay::<mpi>::blueprint::detail --
//-----------------------------------------------------------------------------


//
// Lots of helpers pulled in from Ascent for dealing with
// with mesh blueprint writing and reading. 
// TODO: Could use cleanup.
//
namespace detail
{

class BlueprintTreePathGenerator
{
public:
    //-------------------------------------------------------------------//
    BlueprintTreePathGenerator(const std::string &file_pattern,
                               const std::string &tree_pattern,
                               const std::string &protocol,
                               const Node &mesh_index)
    : m_file_pattern(file_pattern),
      m_tree_pattern(tree_pattern),
      m_protocol(protocol),
      m_mesh_index(mesh_index)
    {

    }

    //-------------------------------------------------------------------//
    ~BlueprintTreePathGenerator()
    {

    }

    //-------------------------------------------------------------------//
    std::string Expand(const std::string pattern,
                       int idx) const
    {
        //
        // Note: This currently only handles format strings:
        // "%05d" "%06d" "%07d"
        //

        std::size_t pattern_idx = pattern.find("%05d");

        if(pattern_idx != std::string::npos)
        {
            char buff[16];
            snprintf(buff,16,"%05d",idx);
            std::string res = pattern;
            res.replace(pattern_idx,4,std::string(buff));
            return res;
        }

        pattern_idx = pattern.find("%06d");

        if(pattern_idx != std::string::npos)
        {
            char buff[16];
            snprintf(buff,16,"%06d",idx);
            std::string res = pattern;
            res.replace(pattern_idx,4,std::string(buff));
            return res;
        }

        pattern_idx = pattern.find("%07d");

        if(pattern_idx != std::string::npos)
        {
            char buff[16];
            snprintf(buff,16,"%07d",idx);
            std::string res = pattern;
            res.replace(pattern_idx,4,std::string(buff));
            return res;
        }
        return pattern;
    }


    //-------------------------------------------------------------------//
    std::string GenerateFilePath(int tree_id) const
    {
        // for now, we only support 1 tree per file.
        int file_id = tree_id;
        return Expand(m_file_pattern,file_id);
    }

    //-------------------------------------------------------------------//
    std::string GenerateTreePath(int tree_id) const
    {
        // the tree path should always end in a /
        std::string res = Expand(m_tree_pattern,tree_id);
        if( (res.size() > 0) && (res[res.size()-1] != '/') )
        {
            res += "/";
        }
        return res;
    }

private:
    std::string m_file_pattern;
    std::string m_tree_pattern;
    std::string m_protocol;
    Node m_mesh_index;

};

bool global_someone_agrees(bool vote
                           CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm mpi_comm))
{
  bool agreement = vote;
#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
  int local_boolean = vote ? 1 : 0;
  int global_boolean;

  MPI_Allreduce((void *)(&local_boolean),
                (void *)(&global_boolean),
                1,
                MPI_INT,
                MPI_SUM,
                mpi_comm);

  if(global_boolean > 0)
  {
    agreement = true;
  }
  else
  {
    agreement = false;
  }
#endif
  return agreement;
}

//
// recalculate domain ids so that we are consistant.
// Assumes that domains are valid
//
void make_domain_ids(conduit::Node &domains
                     CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm mpi_comm))
{
  int num_domains = domains.number_of_children();

  int domain_offset = 0;

#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
  int comm_size = 1;
  int rank = 0;

  MPI_Comm_rank(mpi_comm,&rank);
  MPI_Comm_size(mpi_comm, &comm_size);
  int *domains_per_rank = new int[comm_size];

  MPI_Allgather(&num_domains, 1, MPI_INT, domains_per_rank, 1, MPI_INT, mpi_comm);

  for(int i = 0; i < rank; ++i)
  {
    domain_offset += domains_per_rank[i];
  }
  delete[] domains_per_rank;
#endif

  for(int i = 0; i < num_domains; ++i)
  {
    conduit::Node &dom = domains.child(i);
    dom["state/domain_id"] = domain_offset + i;
  }
}
//
// This expects a single or multi_domain blueprint mesh and will iterate
// through all domains to see if they are valid. Returns true
// if it contains valid data and false if there is no valid
// data.
//
// This is needed because after pipelines, it is possible to
// have no data left in a domain because of something like a
// clip
//
bool clean_mesh(const conduit::Node &data,
                conduit::Node &output
                CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm mpi_comm))
{
  output.reset();
  const int potential_doms = data.number_of_children();
  bool maybe_multi_dom = true;

  if(!data.dtype().is_object() && !data.dtype().is_list())
  {
    maybe_multi_dom = false;
  }

  if(maybe_multi_dom)
  {
    // check all the children for valid domains
    for(int i = 0; i < potential_doms; ++i)
    {
      conduit::Node info;
      const conduit::Node &child = data.child(i);
      bool is_valid = ::conduit::blueprint::mesh::verify(child, info);
      if(is_valid)
      {
        conduit::Node &dest_dom = output.append();
        dest_dom.set_external(child);
      }
    }
  }
  // if there is nothing in the output, lets see if it is a
  // valid single domain
  if(output.number_of_children() == 0)
  {
    // check to see if this is a single valid domain
    conduit::Node info;
    bool is_valid = ::conduit::blueprint::mesh::verify(data, info);
    if(is_valid)
    {
      conduit::Node &dest_dom = output.append();
      dest_dom.set_external(data);
    }
  }

#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
    detail::make_domain_ids(output, mpi_comm);
#else
    detail::make_domain_ids(output);
#endif


  return output.number_of_children() > 0;
}
// mfem needs these special fields so look for them
void check_for_attributes(const conduit::Node &input,
                          std::vector<std::string> &list)
{
  const int num_doms = input.number_of_children();
  std::set<std::string> specials;
  for(int d = 0; d < num_doms; ++d)
  {
    const conduit::Node &dom = input.child(d);
    if(dom.has_path("fields"))
    {
      const conduit::Node &fields = dom["fields"];
      std::vector<std::string> fnames = fields.child_names();
      for(size_t i = 0; i < fnames.size(); ++i)
      {
        if(fnames[i].find("_attribute") != std::string::npos)
        {
          specials.insert(fnames[i]);
        }
      }
    }
  }

  for(auto it = specials.begin(); it != specials.end(); ++it)
  {
    list.push_back(*it);
  }
}

void filter_fields(const conduit::Node &input,
                   conduit::Node &output,
                   std::vector<std::string> fields
                   CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm mpi_comm))
{
  // assume this is multi-domain
  //
  check_for_attributes(input, fields);

  const int num_doms = input.number_of_children();
  for(int d = 0; d < num_doms; ++d)
  {
    const conduit::Node &dom = input.child(d);
    conduit::Node &out_dom = output.append();
    for(size_t f = 0; f < fields.size(); ++f)
    {
      const std::string fname = fields[f];
      if(dom.has_path("fields/" + fname))
      {
        const std::string fpath = "fields/" + fname;
        out_dom[fpath].set_external(dom[fpath]);
        // check for topologies
        const std::string topo = dom[fpath + "/topology"].as_string();
        const std::string tpath = "topologies/" + topo;
        if(!out_dom.has_path(tpath))
        {
          out_dom[tpath].set_external(dom[tpath]);
          if(dom.has_path(tpath + "/grid_function"))
          {
            const std::string gf_name = dom[tpath + "/grid_function"].as_string();
            const std::string gf_path = "fields/" + gf_name;
            out_dom[gf_path].set_external(dom[gf_path]);
          }
          if(dom.has_path(tpath + "/boundary_topology"))
          {
            const std::string bname = dom[tpath + "/boundary_topology"].as_string();
            const std::string bpath = "topologies/" + bname;
            out_dom[bpath].set_external(dom[bpath]);
          }
        }
        // check for coord sets
        const std::string coords = dom[tpath + "/coordset"].as_string();
        const std::string cpath = "coordsets/" + coords;
        if(!out_dom.has_path(cpath))
        {
          out_dom[cpath].set_external(dom[cpath]);
        }
      }
    }
    if(dom.has_path("state"))
    {
      out_dom["state"].set_external(dom["state"]);
    }
  }

  const int num_out_doms = output.number_of_children();
  bool has_data = false;
  // check to see if this resulted in any data
  for(int d = 0; d < num_out_doms; ++d)
  {
    const conduit::Node &dom = output.child(d);
    if(dom.has_path("fields"))
    {
      int fsize = dom["fields"].number_of_children();
      if(fsize != 0)
      {
        has_data = true;
        break;
      }
    }
  }


#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
    has_data = detail::global_someone_agrees(has_data,mpi_comm);
#else
    has_data = detail::global_someone_agrees(has_data);
#endif
  
  if(!has_data)
  {
    CONDUIT_ERROR("Relay: field selection resulted in no data."
                  "This can occur if the fields did not exist "
                  "in the simulation data or if the fields were "
                  "created as a result of a pipeline, but the "
                  "relay extract did not receive the result of "
                  "a pipeline");
  }

}

//---------------------------------------------------------------------------//
std::string
identify_protocol(const std::string &path)
{
    std::string file_path, obj_base;
    conduit::utils::split_file_path(path,
                                    std::string(":"),
                                    file_path,
                                    obj_base);

    std::string file_name_base, file_name_ext;
    conduit::utils::rsplit_string(file_path,
                                  std::string("."),
                                  file_name_ext,
                                  file_name_base);

    std::string io_type = "bin";
    if(file_name_ext.find("blueprint_root") == 0)
    {
        std::string file_name_true_ext = file_name_ext.substr(
            std::string("blueprint_root").length(), file_name_ext.length());

        // TODO: Add support for yaml protocol
        if(file_name_true_ext == "")
        {
            io_type = "json";
        }
        else if(file_name_true_ext == "_hdf5" || file_name_true_ext == "_h5")
        {
            io_type = "hdf5";
        }
        else if(file_name_true_ext == "_silo")
        {
            io_type = "silo";
        }
    }

    return io_type;
}

//-----------------------------------------------------------------------------
void gen_domain_to_file_map(int num_domains,
                            int num_files,
                            Node &out)
{
    int num_domains_per_file = num_domains / num_files;
    int left_overs = num_domains % num_files;

    out["global_domains_per_file"].set(DataType::int32(num_files));
    out["global_domain_offsets"].set(DataType::int32(num_files));
    out["global_domain_to_file"].set(DataType::int32(num_domains));

    int32_array v_domains_per_file = out["global_domains_per_file"].value();
    int32_array v_domains_offsets  = out["global_domain_offsets"].value();
    int32_array v_domain_to_file   = out["global_domain_to_file"].value();

    // setup domains per file
    for(int f=0; f < num_files; f++)
    {
        v_domains_per_file[f] = num_domains_per_file;
        if( f < left_overs)
            v_domains_per_file[f]+=1;
    }

    // prefix sum to calc offsets
    for(int f=0; f < num_files; f++)
    {
        v_domains_offsets[f] = v_domains_per_file[f];
        if(f > 0)
            v_domains_offsets[f] += v_domains_offsets[f-1];
    }

    // do assignment, create simple map
    int f_idx = 0;
    for(int d=0; d < num_domains; d++)
    {
        if(d >= v_domains_offsets[f_idx])
            f_idx++;
        v_domain_to_file[d] = f_idx;
    }
}



//-----------------------------------------------------------------------------
// -- end conduit::relay::<mpi>::io_blueprint::detail --
//-----------------------------------------------------------------------------
};


//-----------------------------------------------------------------------------
// Sig variants of save_mesh
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void save_mesh(const Node &mesh,
               const std::string &path
               CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm mpi_comm))
{
    // empty opts
    Node opts;
#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
    save_mesh(mesh,
              path,
              detail::identify_protocol(path),
              opts,
              mpi_comm);
#else
    save_mesh(mesh,
              path,
              detail::identify_protocol(path),
              opts);
#endif
}

//-----------------------------------------------------------------------------
void save_mesh(const Node &mesh,
               const std::string &path,
               const std::string &protocol
               CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm mpi_comm))
{
    // empty opts
    Node opts;
#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
    save_mesh(mesh,
              path,
              protocol,
              opts, // number of files (-1 == # of domains)
              mpi_comm);
#else
    save_mesh(mesh,
              path,
              protocol,
              opts); // number of files (-1 == # of domains)
#endif
}

//-----------------------------------------------------------------------------
// Main Mesh Blueprint Save, taken from Ascent
//-----------------------------------------------------------------------------
// opts:
//      file_style: "default", "root_only", "multi_file"
//            when # of domains == 1,  "default"   ==> "root_only"
//            else,                    "default"   ==> "multi_file"
//      suffix: "default", "cycle", "none" 
//            when # of domains == 1,  "default"   ==> "off"
//            else,                    "default"   ==> "cycle"
//      number_of_files:  {# of files}
//            when "multi_file":
//                 <= 0, use # of files == # of domains
//                  > 0, # of files == number_of_files
//
void save_mesh(const Node &mesh,
               const std::string &path,
               const std::string &file_protocol,
               const Node &opts
               CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm mpi_comm))
{
    std::string opts_file_style = "default";
    std::string opts_suffix     = "default";
    int         opts_num_files = -1;

    if(opts.has_child("file_style") && opts["file_style"].dtype().is_string())
    {
        opts_file_style = opts["file_style"].as_string();

        if(opts_suffix != "default" && 
           opts_suffix != "root_only" &&
           opts_suffix != "multi_file" )
        {
            CONDUIT_ERROR("save_mesh invalid file_style option: \"" 
                          << opts_file_style << "\"\n"
                          " expected: \"default\", \"root_only\", "
                          "or \"multi_file\"");
        }

    }
    if(opts.has_child("suffix") && opts["suffix"].dtype().is_string())
    {
        opts_suffix = opts["suffix"].as_string();

        if(opts_suffix != "default" && 
           opts_suffix != "cycle" &&
           opts_suffix != "none" )
        {
            CONDUIT_ERROR("save_mesh invalid suffix option: \"" 
                          << opts_suffix << "\"\n"
                          " expected: \"default\", \"cycle\", or \"none\"");
        }
    }

    if(opts.has_child("number_of_files") && opts["number_of_files"].dtype().is_integer())
    {
        opts_num_files = (int) opts["number_of_files"].to_int();
    }

    int num_files = opts_num_files;

    // The assumption here is that everything is multi domain

#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
    // nodes used for MPI comm (share them for many operations)
    Node n_local, n_reduced;
#endif

    // -----------------------------------------------------------
    // make sure some MPI taks has data
    // -----------------------------------------------------------
    Node multi_dom;
#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
    bool is_valid = detail::clean_mesh(mesh, multi_dom, mpi_comm);
#else
    bool is_valid = detail::clean_mesh(mesh, multi_dom);
#endif

    int par_rank = 0;
    int par_size = 1;
    // we may not have any domains so init to max
    int cycle = std::numeric_limits<int>::max();

    int local_boolean = is_valid ? 1 : 0;
    int global_boolean = local_boolean;


#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
    par_rank = relay::mpi::rank(mpi_comm);
    par_size = relay::mpi::size(mpi_comm);

    // reduce to check to see if any valid data exist

    n_local = (int)cycle;
    relay::mpi::sum_all_reduce(n_local,
                               n_reduced,
                               mpi_comm);

    global_boolean = n_reduced.as_int();


#endif

    if(global_boolean == 0)
    {
      CONDUIT_INFO("Blueprint save: no valid data exists. Skipping save");
      return;
    }

    // -----------------------------------------------------------
    // get the number of local domains and the cycle info
    // -----------------------------------------------------------

    int local_num_domains = multi_dom.number_of_children();
    // figure out what cycle we are
    if(local_num_domains > 0 && is_valid)
    {
        Node dom = multi_dom.child(0);
        if(!dom.has_path("state/cycle"))
        {
            if(opts_suffix == "cycle")
            {
                static std::map<std::string,int> counters;
                CONDUIT_INFO("Blueprint save: no 'state/cycle' present."
                            " Defaulting to counter");
                cycle = counters[path];
                counters[path]++;
            }
            else
            {
                opts_suffix = "none";
            }
        }
        else
        {
            cycle = dom["state/cycle"].to_int();
            opts_suffix = "cycle";
        }
    }

#ifdef CONDUIT_RELAY_IO_MPI_ENABLED

    n_local = (int)cycle;

    relay::mpi::min_all_reduce(n_local,
                               n_reduced,
                               mpi_comm);

    cycle = n_reduced.as_int();
#endif

    // find the # of global domains

    int global_num_domains = local_num_domains;

#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
    n_local = local_num_domains;

    relay::mpi::sum_all_reduce(n_local,
                               n_reduced,
                               mpi_comm);

    global_num_domains = n_reduced.as_int();
#endif

    if(global_num_domains == 0)
    {
      if(par_rank == 0)
      {
          CONDUIT_WARN("There no data to save. Doing nothing.");
      }
      return;
    }

    char fmt_buff[64] = {0};
    std::ostringstream oss;

    std::string output_dir = "";

    // default implies multi_file if more than one domain
    if(opts_file_style == "default")
    {
        if( global_num_domains > 1)
        {
            opts_file_style = "multi_file";
        }
        else // other wise, use root only
        {
            opts_file_style = "root_only";
        }
    }

    // ----------------------------------------------------
    // if using multi_file, we need an output dir, create
    // ----------------------------------------------------
    if(opts_file_style == "multi_file")
    {
        // setup the directory
        oss << path;
        // at this point we should only see cycle or none,
        // default has been resolved
        if(opts_suffix == "cycle")
        {
            // TODO: FMT ?
            snprintf(fmt_buff, sizeof(fmt_buff), "%06d",cycle);
            oss << ".cycle_" << fmt_buff;
        }

        output_dir  =  oss.str();
        bool dir_ok = false;

        // let rank zero handle dir creation
        if(par_rank == 0)
        {
            // check of the dir exists
            dir_ok = utils::is_directory(output_dir);
            if(!dir_ok)
            {
                // if not try to let rank zero create it
                dir_ok = utils::create_directory(output_dir);
            }
        }

        // make sure everyone knows if dir creation was successful 

        #ifdef CONDUIT_RELAY_IO_MPI_ENABLED
        // use an mpi sum to check if the dir exists
        n_local = dir_ok ? 1 : 0;

        relay::mpi::sum_all_reduce(n_local,
                                   n_reduced,
                                   mpi_comm);

        dir_ok = (n_reduced.as_int() == 1);
        #endif

        if(!dir_ok)
        {
            CONDUIT_ERROR("Error: failed to create directory " << output_dir);
        }
    }


    // ----------------------------------------------------
    // setup root file name
    // ----------------------------------------------------
    // TODO FMT
    oss.str("");
    oss << path;
    // at this point we should only see cycle or none, default
    // has been resolved
    if(opts_suffix == "cycle")
    {
        snprintf(fmt_buff, sizeof(fmt_buff), "%06d",cycle);
        oss << ".cycle_" << fmt_buff;
    }
    oss << ".root";

    std::string root_filename = oss.str();

    // zero or negative (default cases), use one file per domain
    if(num_files <= 0)
    {
        num_files = global_num_domains;
    }

    // if global domains > num_files, warn and use one file per domain
    if(global_num_domains < num_files)
    {
        CONDUIT_INFO("Requested more files than actual domains, "
                     "writing one file per domain");
        num_files = global_num_domains;
    }

    // ------------------------
    // default has been resolved, we need to just handle:
    // root_only, multi_file
    if(opts_file_style == "root_only")
    {
        // write out local domains, since all tasks will
        // write to single file in this case, we need baton.

        relay::io::IOHandle hnd;

        for(int current_writer=0; current_writer < par_size; current_writer++)
        {
            if(par_rank == current_writer)
            {
                for(int i = 0; i < local_num_domains; ++i)
                {
                    if(!hnd.is_open())
                    {
                        hnd.open(root_filename,file_protocol);
                    }

                    const Node &dom = multi_dom.child(i);
                    std::string mesh_path = "";

                    if(global_num_domains == 1) // no domain prefix, write to "mesh"
                    {
                        mesh_path = "mesh";
                    }
                    else
                    {
                        uint64 domain = dom["state/domain_id"].to_uint64();
                        // TODO FMT
                        snprintf(fmt_buff, sizeof(fmt_buff), "%06llu",domain);
                        oss.str("");
                        oss << "domain_" << fmt_buff;
                        mesh_path = oss.str();
                    }
                    hnd.write(dom,mesh_path);
                    // std::string output_file  = conduit::utils::join_file_path(output_dir,oss.str());
                    // relay::io::save(dom, output_file);
                }
            }

    #ifdef CONDUIT_RELAY_IO_MPI_ENABLED
        MPI_Barrier(mpi_comm);
    #endif

        }
        // // if we have an open handle, close it here
        // if(!hnd.is_open())
        // {
        //     hnd.close();
        // }
    }
    else if(global_num_domains == num_files)
    {
        // write out each domain
        for(int i = 0; i < local_num_domains; ++i)
        {
            const Node &dom = multi_dom.child(i);
            uint64 domain = dom["state/domain_id"].to_uint64();

            snprintf(fmt_buff, sizeof(fmt_buff), "%06llu",domain);
            oss.str("");
            oss << "domain_" << fmt_buff << "." << file_protocol;
            std::string output_file  = conduit::utils::join_file_path(output_dir,oss.str());
            relay::io::save(dom, output_file);
        }
    }
    else // more complex case
    {
        //
        // recall: we have re-labeled domain ids from 0 - > N-1, however
        // some mpi tasks may have no data.
        //

        // books we keep:
        Node books;
        books["local_domain_to_file"].set(DataType::int32(local_num_domains));
        books["local_domain_status"].set(DataType::int32(local_num_domains));
        books["local_file_batons"].set(DataType::int32(num_files));
        books["global_file_batons"].set(DataType::int32(num_files));

        // local # of domains
        int32_array local_domain_to_file = books["local_domain_to_file"].value();
        int32_array local_domain_status  = books["local_domain_status"].value();
        // num total files
        int32_array local_file_batons    = books["local_file_batons"].value();
        int32_array global_file_batons   = books["global_file_batons"].value();

        Node d2f_map;
        detail::gen_domain_to_file_map(global_num_domains,
                                       num_files,
                                       books);

        int32_array global_d2f = books["global_domain_to_file"].value();

        // init our local map and status array
        for(int d = 0; d < local_num_domains; ++d)
        {
            const Node &dom = multi_dom.child(d);
            uint64 domain = dom["state/domain_id"].to_uint64();
            // local domain index to file map
            local_domain_to_file[d] = global_d2f[domain];
            local_domain_status[d] = 1; // pending (1), vs done (0)
        }

        //
        // Round and round we go, will we deadlock I believe no :-)
        //
        // Here is how this works:
        //  At each round, if a rank has domains pending to write to a file,
        //  we put the rank id in the local file_batons vec.
        //  This vec is then mpi max'ed, and the highest rank
        //  that needs access to each file will write this round.
        //
        //  When a rank does not need to write to a file, we
        //  put -1 for this rank.
        //
        //  During each round, max of # files writers are participating
        //
        //  We are done when the mpi max of the batons is -1 for all files.
        //

        bool another_twirl = true;
        int twirls = 0;
        while(another_twirl)
        {
            // update baton requests
            for(int f = 0; f < num_files; ++f)
            {
                for(int d = 0; d < local_num_domains; ++d)
                {
                    if(local_domain_status[d] == 1)
                        local_file_batons[f] = par_rank;
                    else
                        local_file_batons[f] = -1;
                }
            }

            // mpi max file batons array
            #ifdef CONDUIT_RELAY_IO_MPI_ENABLED
                mpi::max_all_reduce(books["local_file_batons"],
                                    books["global_file_batons"],
                                    mpi_comm);
            #else
                global_file_batons.set(local_file_batons);
            #endif

            // we now have valid batons (global_file_batons)
            for(int f = 0; f < num_files; ++f)
            {
                // check if this rank has the global baton for this file
                if( global_file_batons[f] == par_rank )
                {
                    // check the domains this rank has pending
                    for(int d = 0; d < local_num_domains; ++d)
                    {
                        // reuse this handle for all domains in the file
                        relay::io::IOHandle hnd;
                        if(local_domain_status[d] == 1 &&  // pending
                           local_domain_to_file[d] == f) // destined for this file
                        {
                            // now is the time to write!
                            // pattern is:
                            //  file_%06llu.{protocol}:/domain_%06llu/...
                            const Node &dom = multi_dom.child(d);
                            uint64 domain_id = dom["state/domain_id"].to_uint64();
                            // construct file name
                            snprintf(fmt_buff, sizeof(fmt_buff), "%06d",f);
                            oss.str("");
                            oss << "file_" << fmt_buff << "." << file_protocol;
                            std::string file_name = oss.str();
                            oss.str("");
                            // and now domain id
                            snprintf(fmt_buff, sizeof(fmt_buff), "%06llu",domain_id);
                            oss << "domain_" << fmt_buff;

                            std::string path = oss.str();
                            std::string output_file = conduit::utils::join_file_path(output_dir,
                                                                                     file_name);

                            if(!hnd.is_open())
                            {
                                hnd.open(output_file);
                            }

                            hnd.write(dom,path);
                            // CONDUIT_INFO("rank " << par_rank << " output_file"
                            //              << output_file << " path " << path);

                            // update status, we are done with this doman
                            local_domain_status[d] = 0;
                        }
                    }
                }
            }

            // If you  need to debug the baton alog:
            // std::cout << "[" << par_rank << "] "
            //              << " twirls: " << twirls
            //              << " details\n"
            //              << books.to_yaml();

            // check if we have another round
            // stop when all batons are -1
            another_twirl = false;
            for(int f = 0; f < num_files && !another_twirl; ++f)
            {
                // if any entry is not -1, we still have more work to do
                if(global_file_batons[f] != -1)
                {
                    another_twirl = true;
                    twirls++;
                }
            }
        }
    }

    int root_file_writer = 0;
    if(local_num_domains == 0)
    {
      root_file_writer = -1;
    }
#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
    // Rank 0 could have an empty domain, so we have to check
    // to find someone with a data set to write out the root file.
    Node out;
    out = local_num_domains;
    Node rcv;

    mpi::all_gather_using_schema(out, rcv, mpi_comm);
    root_file_writer = -1;
    int* res_ptr = (int*)rcv.data_ptr();
    for(int i = 0; i < par_size; ++i)
    {
        if(res_ptr[i] != 0)
        {
            root_file_writer = i;
            break;
        }
    }

    MPI_Barrier(mpi_comm);
#endif

    if(root_file_writer == -1)
    {
        // this should not happen. global doms is already 0
        CONDUIT_WARN("Relay: there are no domains to write out");
    }

    // let rank zero write out the root file
    if(par_rank == root_file_writer)
    {
        std::string output_dir_base, output_dir_path;
        conduit::utils::rsplit_file_path(output_dir,
                                         output_dir_base,
                                         output_dir_path);

        std::string output_tree_pattern;
        std::string output_file_pattern;

        if(opts_file_style == "root_only")
        {
            output_file_pattern = root_filename;
            if(global_num_domains == 1)
            {
                output_tree_pattern = "/";
            }
            else
            {
                output_tree_pattern = "/domain_%06d/";
            }
        }
        else if(global_num_domains == num_files)
        {
            output_tree_pattern = "/";
            output_file_pattern = conduit::utils::join_file_path(output_dir_base,
                                                                 "domain_%06d." + file_protocol);
        }
        else
        {
            output_tree_pattern = "/domain_%06d";
            output_file_pattern = conduit::utils::join_file_path(output_dir_base,
                                                                 "file_%06d." + file_protocol);
        }

        Node root;
        Node &bp_idx = root["blueprint_index"];

        // TODO: Use MPI ver vs providing the domains?
        ::conduit::blueprint::mesh::generate_index(multi_dom.child(0),
                                                   "",
                                                   global_num_domains,
                                                   bp_idx["mesh"]);

        // work around conduit and manually add state fields
        if(multi_dom.child(0).has_path("state/cycle"))
        {
          bp_idx["mesh/state/cycle"] = multi_dom.child(0)["state/cycle"].to_int32();
        }

        if(multi_dom.child(0).has_path("state/time"))
        {
          bp_idx["mesh/state/time"] = multi_dom.child(0)["state/time"].to_double();
        }

        root["protocol/name"]    = file_protocol;
        root["protocol/version"] = CONDUIT_VERSION;

        root["number_of_files"]  = num_files;
        root["number_of_trees"]  = global_num_domains;
        // TODO: make sure this is relative
        root["file_pattern"]     = output_file_pattern;
        root["tree_pattern"]     = output_tree_pattern;

        relay::io::IOHandle hnd;
        hnd.open(root_filename, file_protocol);
        hnd.write(root);
        hnd.close();
    }
}

//-----------------------------------------------------------------------------
void load_mesh(const std::string &root_file_path,
               Node &mesh
               CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm mpi_comm))
{
    Node opts;
#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
    load_mesh(root_file_path,
              opts,
              mesh,
              mpi_comm);
#else
    load_mesh(root_file_path,
              opts,
              mesh);
#endif
}

//-----------------------------------------------------------------------------
void load_mesh(const std::string &root_file_path,
               const Node &opts,
               Node &mesh
               CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm mpi_comm))
{
    std::string root_fname = root_file_path;

    // read the root file, it can be either json or hdf5

    // assume hdf5, but check for json file
    std::string root_protocol = "hdf5";
    char buff[5] = {0,0,0,0,0};

    // heuristic, if json, we expect to see "{" in the first 5 chars of the file.
    std::ifstream ifs;
    ifs.open(root_fname.c_str());
    if(!ifs.is_open())
    {
        CONDUIT_ERROR("failed to open root file: " << root_fname);
    }

    if(!ifs.read((char *)buff,5))
    {
        CONDUIT_ERROR("failed to read starting bytes from root file: " << root_fname);
    }
    ifs.close();

    std::string test_str(buff);

    if(test_str.find("{") != std::string::npos)
    {
       root_protocol = "json";
    }

    Node root_node;
    relay::io::load(root_fname, root_protocol, root_node);


    if(!root_node.has_child("file_pattern"))
    {
        CONDUIT_ERROR("Root file missing 'file_pattern'");
    }

    if(!root_node.has_child("blueprint_index"))
    {
        CONDUIT_ERROR("Root file missing 'blueprint_index'");
    }


    std::string mesh_name ="";
    if(opts.has_child("mesh_name") && opts["mesh_name"].dtype().is_string())
    {
        mesh_name = opts["mesh_name"].as_string();
    }

    if(mesh_name.empty())
    {
        NodeConstIterator itr = root_node["blueprint_index"].children();
        itr.next();
        mesh_name = itr.name();
    }

    if(!root_node["blueprint_index"].has_child(mesh_name))
    {
        // bad name, construct an error message that
        // displays the valid options
        std::ostringstream oss;
        oss << "Mesh named '" << mesh_name << "' "
            << " not found in " 
            << root_file_path
            << std::endl
            << " Mesh names found blueprint index: " 
            << std::endl;
        NodeConstIterator itr = root_node["blueprint_index"].children();
        while(itr.has_next())
        {
            itr.next();
            oss << " " << itr.name();
            oss << std::endl;
        }

        CONDUIT_ERROR(oss.str());
    }

    // make sure we have a valid bp index
    Node verify_info;
    const Node &mesh_index = root_node["blueprint_index"][mesh_name];
    if( !::conduit::blueprint::mesh::index::verify(mesh_index,
                                                   verify_info[mesh_name]))
    {
        CONDUIT_ERROR("Mesh Blueprint index verify failed" << std::endl
                      << verify_info.to_json());
    }

    std::string data_protocol = "hdf5";

    if(root_node.has_child("protocol"))
    {
        data_protocol = root_node["protocol/name"].as_string();
    }

    // read all domains for given mesh
    int num_domains = root_node["number_of_trees"].to_int();
    detail::BlueprintTreePathGenerator gen(root_node["file_pattern"].as_string(),
                                           root_node["tree_pattern"].as_string(),
                                           data_protocol,
                                           mesh_index);

    std::ostringstream oss;
    int domain_start = 0;
    int domain_end = num_domains;

#if CONDUIT_RELAY_IO_MPI_ENABLED
    int rank = relay::mpi::rank(mpi_comm);
    int total_size = relay::mpi::size(mpi_comm);

    int read_size = num_domains / total_size;
    int rem = num_domains % total_size;
    if(rank < rem)
    {
        read_size++;
    }

    conduit::Node n_read_size;
    conduit::Node n_doms_per_rank;

    n_read_size.set_int32(read_size);

    relay::mpi::all_gather_using_schema(n_read_size,
                                        n_doms_per_rank,
                                        mpi_comm);
    int *counts = (int*)n_doms_per_rank.data_ptr();

    int rank_offset = 0;
    for(int i = 0; i < rank; ++i)
    {
        rank_offset += counts[i];
    }

    domain_start = rank_offset;
    domain_end = rank_offset + read_size;
#endif

    if(data_protocol == "sidre_hdf5")
    {
        relay::io::IOHandle hnd;
        hnd.open(root_fname,"sidre_hdf5");
        for(int i = domain_start ; i < domain_end; i++)
        {
            std::ostringstream oss;
            oss << i << "/" << mesh_name;
            hnd.read(oss.str(),mesh);
        }
    }
    else
    {
        relay::io::IOHandle hnd;
        for(int i = domain_start ; i < domain_end; i++)
        {
        // TODO FMT
        char domain_fmt_buff[64];
        snprintf(domain_fmt_buff, sizeof(domain_fmt_buff), "%06d",i);
        oss.str("");
        oss << "domain_" << std::string(domain_fmt_buff);

        std::string current, next;
        utils::rsplit_file_path (root_fname, current, next);
        std::string domain_file = utils::join_path(next, gen.GenerateFilePath(i));
        // also need the tree path

        hnd.open(domain_file, data_protocol);
        hnd.read(gen.GenerateTreePath(i), mesh[oss.str()]);
        }
    }
    
}


//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end conduit::relay::io::<mpi>::blueprint --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit::relay::io --
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// -- begin conduit::relay::<mpi>::io_blueprint -- (DEPRECATED)
//-----------------------------------------------------------------------------
namespace io_blueprint
{

//---------------------------------------------------------------------------//
// DEPRECATED
//---------------------------------------------------------------------------//
void
save(const Node &mesh,
     const std::string &path
     CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm comm))
{
#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
    save(mesh,
         path,
         relay::mpi::io::blueprint::detail::identify_protocol(path),
         comm);
#else
    save(mesh,
         path,
         relay::io::blueprint::detail::identify_protocol(path));
#endif
}

//---------------------------------------------------------------------------//
// DEPRECATED
//---------------------------------------------------------------------------//
void
save(const Node &mesh,
     const std::string &path,
     const std::string &protocol
     CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm /*comm*/))
{
    // TODO: Add support for yaml protocol
    Node info;
    if(protocol != "json" && protocol != "hdf5")
    {
        CONDUIT_ERROR("Blueprint I/O doesn't support '" << protocol << "' outputs; "
                      "output type must be 'blueprint_root' (JSON) or 'blueprint_root_hdf5' (HDF5): " <<
                      "Failed to save mesh to path " << path);
    }
    if(!::conduit::blueprint::mesh::verify(mesh, info))
    {
        CONDUIT_ERROR("Given node isn't a valid Blueprint mesh: " <<
                      "Failed to save mesh to path " << path);
    }

    // NOTE(JRC): The code below is used in lieu of `blueprint::mesh::to_multi_domain`
    // because the official Blueprint function produces results that are incompatible
    // with HDF5 outputs (because they include Conduit lists instead of dictionaries).
    Node index;
    if(::conduit::blueprint::mesh::is_multi_domain(mesh))
    {
        index["data"].set_external(mesh);
    }
    else
    {
        index["data/mesh"].set_external(mesh);
    }

    Node &bpindex = index["blueprint_index"];
    {
        NodeConstIterator domain_iter = index["data"].children();
        while(domain_iter.has_next())
        {
            const Node &domain = domain_iter.next();
            const std::string domain_name = domain_iter.name();

            // NOTE: Skip all domains containing one or more mixed-shape topologies
            // because this type of mesh isn't fully supported yet.
            bool is_domain_index_valid = true;
            NodeConstIterator topo_iter = domain["topologies"].children();
            while(topo_iter.has_next())
            {
                const Node &topo = topo_iter.next();
                is_domain_index_valid &= (
                    !::conduit::blueprint::mesh::topology::unstructured::verify(topo, info) ||
                    !topo["elements"].has_child("element_types"));
            }

            if(is_domain_index_valid)
            {
                ::conduit::blueprint::mesh::generate_index(
                    domain,domain_name,1,bpindex[domain_name]);
            }
        }
    }

    if(bpindex.number_of_children() == 0)
    {
        CONDUIT_INFO("No valid domains in given Blueprint mesh: " <<
                     "Skipping save of mesh to path " << path);
    }
    else
    {
        index["protocol/name"].set(protocol);
        index["protocol/version"].set(CONDUIT_VERSION);

        index["number_of_files"].set(1);
        index["number_of_trees"].set(1);
        index["file_pattern"].set(path);
        index["tree_pattern"].set((protocol == "hdf5") ? "data/" : "data");

        relay::io::save(index,path,protocol);
    }
}



//-----------------------------------------------------------------------------
}
//-----------------------------------------------------------------------------
// -- end conduit::relay::<mpi>::io_blueprint --
//-----------------------------------------------------------------------------



#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
}
//-----------------------------------------------------------------------------
// -- end conduit::relay::mpi --
//-----------------------------------------------------------------------------
#endif

}
//-----------------------------------------------------------------------------
// -- end conduit::relay --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------
