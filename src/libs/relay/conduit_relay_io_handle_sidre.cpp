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
/// file: conduit_relay_io_handle_sidre.cpp
///
//-----------------------------------------------------------------------------

#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
    // FIXME:
    #include "conduit_relay_io_handle.hpp"
#else
    #include "conduit_relay_io_handle.hpp"
#endif

#include "conduit_relay_io.hpp"
#include "conduit_relay_io_handle.hpp"
#include "conduit_relay_io_handle_sidre.hpp"


//-----------------------------------------------------------------------------
// standard lib includes
//-----------------------------------------------------------------------------

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
// -- begin conduit::relay::<mpi>::io --
//-----------------------------------------------------------------------------
namespace io
{

//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// SidreHandle Implementation 
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
SidreHandle::SidreHandle(const std::string &path,
                         const std::string &protocol,
                         const Node &options)
: HandleInterface(path,protocol,options)
{
    // empty
}

//-----------------------------------------------------------------------------
SidreHandle::~SidreHandle()
{
    close();
}

//-----------------------------------------------------------------------------
void 
SidreHandle::open()
{
    close();

    // call base class method, which does final sanity checks
    // and processes standard options (mode = "rw", etc)
    HandleInterface::open();

    if( open_mode() == "w" )
        CONDUIT_ERROR("SidreHandle does not support write mode "
                      "(open_mode = 'w')");

    // the root file will be what is passed to path
    m_root_file = path();

    if( !utils::is_file( m_root_file ) )
        CONDUIT_ERROR("Invalid sidre root file: " << m_root_file);

    std::string m_root_protocol = detect_root_protocol();

    m_root_handle.open(m_root_file, m_root_protocol);

    // check for standard sidre root file entries, we need:
    // file_pattern, tree_pattern, number_of_trees, number_of_files, protocol

    if(!m_root_handle.has_path("file_pattern"))
        CONDUIT_ERROR("Sidre root file missing entry: file_pattern")

    if(!m_root_handle.has_path("tree_pattern"))
        CONDUIT_ERROR("Sidre root file missing entry: tree_pattern")

    if(!m_root_handle.has_path("number_of_trees"))
        CONDUIT_ERROR("Sidre root file missing entry: number_of_trees")

    if(!m_root_handle.has_path("number_of_files"))
        CONDUIT_ERROR("Sidre root file missing entry: number_of_files")

    if(!m_root_handle.has_path("protocol"))
        CONDUIT_ERROR("Sidre root file missing entry: protocol")

    // read the standard entries
    Node root_info; 
    m_root_handle.read("file_pattern",root_info["file_pattern"]);
    m_root_handle.read("tree_pattern",root_info["tree_pattern"]);
    m_root_handle.read("number_of_trees",root_info["number_of_trees"]);
    m_root_handle.read("number_of_files",root_info["number_of_files"]);
    m_root_handle.read("protocol", root_info["protocol"]);

    m_num_trees = root_info["number_of_trees"].to_int();
    m_num_files = root_info["number_of_files"].to_int();

    m_file_pattern = root_info["file_pattern"].as_string();
    m_tree_pattern = root_info["tree_pattern"].as_string();

    // read protocol/name to obtain correct protocol for data
    // we expect a string like the following:
    //    sidre_zzz
    // where zzz is that value we want as the file protocol

    m_file_protocol = root_info["protocol/name"].as_string();
    std::string curr,next;
    utils::split_string(m_file_protocol,"_",curr,next);
    m_file_protocol = next;

    std::cout << "file proto = " << m_file_protocol << std::endl;
    m_open = true;
}


//-----------------------------------------------------------------------------
bool
SidreHandle::is_open() const
{
    return m_open;
}

//-----------------------------------------------------------------------------
void 
SidreHandle::read(Node &node)
{
    if( open_mode() == "w")
    {
        CONDUIT_ERROR("IOHandle: cannot read, handle is write only"
                      " (mode = 'w')");
    }

    std::vector<std::string> child_names;
    list_child_names(child_names);

    for(size_t i=0;i<child_names.size();i++)
    {
        read(child_names[i],node[child_names[i]]);
    }
}

//-----------------------------------------------------------------------------
void 
SidreHandle::read(const std::string &path,
                 Node &node)
{
    if( open_mode() == "w")
    {
        CONDUIT_ERROR("IOHandle: cannot read, handle is write only"
                      " (mode = 'w')");
    }

    // if blank path or "/", use other method and early exist.
    if(path.empty() || path == "/")
    {
        read(node);
        return;
    }

    // split the path to find out what we need to read
    // "root" =>> read_from_root(path,node)
    // tree_id (number) ==> read_from_sidre_tree(tree_id,path)

    
    std::string p_first;
    std::string p_next;
    utils::split_path(path,p_first,p_next);

    bool is_number = true;
    if(p_first == "root")
    {
        read_from_root(p_next,node);
    }
    else if(is_number) // TODO: We are assuming this is a number, yikes!
    {
        int tree_id = -1;
        std::istringstream iss(p_first);
        iss >> tree_id;
        read_from_sidre_tree(tree_id,p_next,node);
    }
    else
    {
        // TODO BAD PATH
    }
}

//-----------------------------------------------------------------------------
void 
SidreHandle::write(const Node & /*node*/) // node is unused
{
    // throw an error if we opened in "r" mode
    if( open_mode() == "r")
    {
        CONDUIT_ERROR("IOHandle: cannot write, handle is read only"
                      " (mode = 'r')");
    }

    // not supported yet, so throw a fatal error
    CONDUIT_ERROR("IOHandle: sidre write support not implemented");
}


//-----------------------------------------------------------------------------
void 
SidreHandle::write(const Node & /*node*/, // node is unused
                   const std::string & /*path*/ ) // path is unused
{
    // throw an error if we opened in "r" mode
    if( open_mode() == "r")
    {
        CONDUIT_ERROR("IOHandle: cannot write, handle is read only"
                      " (mode = 'r')");
    }

    // not supported yet, so throw a fatal error
    CONDUIT_ERROR("IOHandle: sidre write support not implemented");

}

//-----------------------------------------------------------------------------
void
SidreHandle::list_child_names(std::vector<std::string> &res) const
{
    if( open_mode() == "w")
    {
        CONDUIT_ERROR("IOHandle: cannot list_child_names, handle is write only"
                      " (mode = 'w')");
    }

    // root case of the file name populate with root + tree_ids

    res.clear();
    res.push_back("root");

    std::ostringstream oss;
    for(int i=0;i<m_num_trees;i++)
    {
        oss.str("");
        oss << i;
        res.push_back(oss.str());
    }
}

//-----------------------------------------------------------------------------
void
SidreHandle::list_child_names(const std::string &path,
                             std::vector<std::string> &res) const
{
    if( open_mode() == "w")
    {
        CONDUIT_ERROR("IOHandle: cannot list_child_names, handle is write only"
                      " (mode = 'w')");
    }

    std::string p_first;
    std::string p_next;
    utils::split_path(path,p_first,p_next);

    if(p_first == "root")
    {
        m_root_handle.list_child_names(p_next,res);
    }
    else // TODO: We are assuming this is a number, yikes!
    {
        int tree_id = -1;
        std::istringstream iss(p_first);
        iss >> tree_id;
        //  TODO: GET FROM SIDRE META!
    }
}

//-----------------------------------------------------------------------------
void 
SidreHandle::remove(const std::string &/*path*/) // path is unused
{
    // throw an error if we opened in "r" mode
    if( open_mode() == "r")
    {
        CONDUIT_ERROR("IOHandle: cannot remove path, handle is read only"
                      " (mode = 'r')");
    }

    // not supported yet, so throw a fatal error
    CONDUIT_ERROR("IOHandle: sidre write support not implemented");
}

//-----------------------------------------------------------------------------
bool 
SidreHandle::has_path(const std::string &path) const
{
    if( open_mode() == "w")
    {
        CONDUIT_ERROR("IOHandle: cannot call has_path, handle is write only"
                      " (mode = 'w')");
    }

    bool res = false;
    std::string p_first;
    std::string p_next;
    utils::split_path(path,p_first,p_next);

    if(p_first == "root")
    {
        res = m_root_handle.has_path(p_next);
    }
    else // TODO: We are assuming this is a number, yikes!
    {
        int tree_id = -1;
        std::istringstream iss(p_first);
        iss >> tree_id;
        //  GET FROM SIDRE META!
    }

    return res;
}


//-----------------------------------------------------------------------------
void 
SidreHandle::close()
{
    m_open = false;
    m_root_handle.close();

    // TODO: clear should call close when handles are destructed ...
    // double check
    // for( std::map<int,IOHandle>::iterator itr = m_file_handles.begin();
    //      itr != m_file_handles.end();
    //      itr++)
    // {
    //     itr->second.close();
    // }
    m_file_handles.clear();
    m_sidre_meta.clear();
}

//-----------------------------------------------------------------------------
// Helpers
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
std::string 
SidreHandle::root_file_directory() const
{
    std::string curr, next;
    utils::rsplit_file_path(m_root_file, curr, next);
    return next;
}


//-------------------------------------------------------------------//
std::string 
SidreHandle::detect_root_protocol() const
{
    //
    // TODO: detect root file type -- could be hdf5, json, or yaml
    //
    return "hdf5";
}

//-------------------------------------------------------------------//
// adapted from VisIt, avtBlueprintTreeCache
//              ascent, hola, BlueprintTreePathGenerator
std::string 
SidreHandle::expand_pattern(const std::string pattern,
                            int idx) const
{
    //
    // Note: This currently handles format strings:
    // "%d" "%02d" "%03d" "%04d" "%05d" "%06d" "%07d" "%08d" "%09d"
    //

    std::size_t pattern_idx = pattern.find("%d");
    char buff[16] = {0};

    if(pattern_idx != std::string::npos)
    {
        snprintf(buff,16,"%d",idx);
        std::string res = pattern;
        res.replace(pattern_idx,2,std::string(buff));
        return res;
    }

    std::ostringstream oss;
    for(int i=2;i<9;i++)
    {
        oss.str("");
        oss << "%0"  << i << "d";
        pattern_idx = pattern.find(oss.str());
        if(pattern_idx != std::string::npos)
        {
            snprintf(buff,16,oss.str().c_str(),idx);
            std::string res = pattern;
            res.replace(pattern_idx,4,std::string(buff));
            return res;
        }
    }

    return pattern;
}

//-------------------------------------------------------------------//
// adapted from VisIt, avtBlueprintTreeCache
int
SidreHandle::generate_file_id_for_tree(int tree_id) const
{
    int file_id = -1;

    if(m_num_trees == m_num_files)
    {
        file_id = tree_id;
    }
    else if(m_num_files == 1)
    {
        file_id = 0;
    }
    else
    {
        // TODO: Cache the d2f_map
        Node d2f_map;
        generate_domain_to_file_map(m_num_trees,
                                    m_num_files,
                                    d2f_map);
        int32_array v_domain_to_file = d2f_map["global_domain_to_file"].value();
        file_id = v_domain_to_file[tree_id];
    }

    return file_id;
}

//-------------------------------------------------------------------//
// adapted from VisIt, avtBlueprintTreeCache
std::string
SidreHandle::generate_file_path(int tree_id) const
{
    // TODO: Map for tree_ids to file_ids?
    int file_id = generate_file_id_for_tree(tree_id);
    return utils::join_path(root_file_directory(),
                            expand_pattern(m_file_pattern,file_id));
}

//-------------------------------------------------------------------//
// adapted from VisIt, avtBlueprintTreeCache
std::string
SidreHandle::generate_tree_path(int tree_id) const
{
    // the tree path should always end in a /
    std::string res = expand_pattern(m_tree_pattern,tree_id);
    if( (res.size() > 0) && (res[res.size()-1] != '/') )
    {
        res += "/";
    }
    return res;
}

//-----------------------------------------------------------------------------
// This uses a mapping scheme created by ascent + conduit
// Note: We will support explicit maps from the bp index in the future.
// TODO: We should cache this, not regen every fetch
//-----------------------------------------------------------------------------
// adapted from VisIt, avtBlueprintTreeCache
//-----------------------------------------------------------------------------
void
SidreHandle::generate_domain_to_file_map(int num_domains,
                                         int num_files,
                                         Node &out) const
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

//---------------------------------------------------------------------------//
// Helper that expands a sidre groups's meta path to the expected file path 
//---------------------------------------------------------------------------//
std::string
SidreHandle::generate_sidre_meta_group_path(const std::string &tree_path) const
{
    // sidre/groups/ga/groups/gb/groups/gc/.../groups/gn

    std::ostringstream oss;

    std::string t_path  = tree_path;
    std::string t_curr;
    std::string t_next;

    while(t_path != "")
    {
        conduit::utils::split_path(t_path,
                                   t_curr,
                                   t_next);
                                   
        oss << "groups/" << t_curr;
        if( t_next != "")
        {
            oss << "/";
        }

        t_path = t_next;

    }

    return oss.str();

}

//---------------------------------------------------------------------------//
// Helper that expands a sidre view's meta path to the expected file path
//---------------------------------------------------------------------------//
std::string
SidreHandle::generate_sidre_meta_view_path(const std::string &tree_path) const
{
    // sidre/groups/ga/groups/gb/groups/gc/.../views/gn

    std::ostringstream oss;

    std::string t_path  = tree_path;
    std::string t_curr;
    std::string t_next;

    while(t_path != "")
    {
        conduit::utils::split_path(t_path,
                                   t_curr,
                                   t_next);
        if(t_next == "")
        {
            oss << "/views/" << t_curr;
        }
        else
        {
            oss << "groups/" << t_curr << "/";
        }
        t_path = t_next;
    }

    return oss.str();
}

//----------------------------------------------------------------------------/
void
SidreHandle::load_sidre_tree(Node &sidre_meta,
                             int tree_id,
                             const std::string &tree_path,
                             const std::string &curr_path,
                             Node &out)
{
    std::cout << "load_sidre_tree w/ meta" << tree_id << 
              " " << tree_path << " " << curr_path << std::endl;
    // we want to pull out a sub-tree of the sidre group hierarchy 
    //
    // descend down to "tree_path" in sidre meta

    std::string tree_curr;
    std::string tree_next;
    conduit::utils::split_path(tree_path,tree_curr,tree_next);

    if( sidre_meta["groups"].has_path(tree_curr) )
    {
        // BP_PLUGIN_INFO(curr_path << tree_curr << " is a group");
        if(tree_next.size() == 0)
        { 
            // we have the correct sidre meta node, now read
            load_sidre_group(sidre_meta["groups"][tree_curr],
                             tree_id,
                             curr_path + tree_curr  + "/",
                             out);
        }
        else // keep descending 
        {
            load_sidre_tree(sidre_meta["groups"][tree_curr],
                            tree_id,
                            tree_next,
                            curr_path + tree_curr  + "/",
                            out);
        }
    }
    else if( sidre_meta["view"].has_path(tree_curr) )
    {
        // BP_PLUGIN_INFO(curr_path << tree_curr << " is a view");
        if(tree_next.size() != 0)
        {
            CONDUIT_ERROR("Sidre path extends beyond sidre view, "
                          "however Sidre views are leaves.");
        }
        else
        {
            load_sidre_view(sidre_meta["view"][tree_curr],
                            tree_id,
                            curr_path + tree_curr  + "/",
                            out);
        }
    }
    else
    {
        CONDUIT_ERROR("sidre tree_id " << tree_id << """ "" does not exist");
        //error: );
    }
}

//----------------------------------------------------------------------------/
void
SidreHandle::load_sidre_group(Node &sidre_meta,
                              int tree_id,
                              const std::string &group_path,
                              Node &out)
{
    std::cout << "load_sidre_group " << tree_id << 
              " " << group_path << std::endl;
    
    // load this group's children groups and views
    NodeIterator g_itr = sidre_meta["groups"].children();
    while(g_itr.has_next())
    {
        Node &g = g_itr.next();
        std::string g_name = g_itr.name();
        //BP_PLUGIN_INFO("loading " << group_path << g_name << " as group");
        std::string cld_path = group_path + g_name;
        load_sidre_group(g,
                         tree_id,
                         cld_path + "/",
                         out[g_name]);
    }

    NodeIterator v_itr = sidre_meta["views"].children();
    while(v_itr.has_next())
    {
        Node &v = v_itr.next();
        std::string v_name = v_itr.name();
        // BP_PLUGIN_INFO("loading " << group_path << v_name << " as view");
        std::string cld_path = group_path + v_name;
        load_sidre_view(v,
                       tree_id,
                       cld_path,
                       out[v_name]);
    }
}


//----------------------------------------------------------------------------/
void
SidreHandle::load_sidre_view(Node &sidre_meta_view,
                             int tree_id,
                             const std::string &view_path,
                             Node &out)
{
    std::cout << "load_sidre_view " << tree_id << 
              " " << view_path << std::endl;

    // view load cases:
    //   the view is a scalar or string
    //     simply copy the "value" from the meta view
    //
    //   the view is attached to a buffer
    //     in this case we need to get the info about the buffer the view is 
    //     attached to and read the proper slab of that buffer's hdf5 dataset
    //     into a new compact node.
    //
    //   the view is has external data
    //     for this case we can follow the "tree_path" in the sidre external
    //     data tree, and fetch the hdf5 dataset that was written there.
    //

    std::string view_state = sidre_meta_view["state"].as_string();

    if( view_state == "STRING")
    {
        //BP_PLUGIN_INFO("loading " << view_path << " as sidre string view");
        out.set(sidre_meta_view["value"]);
    }
    else if(view_state == "SCALAR")
    {
        // BP_PLUGIN_INFO("loading " << view_path << " as sidre scalar view");
        out.set(sidre_meta_view["value"]);
    }
    else if( view_state == "BUFFER" )
    {
        // BP_PLUGIN_INFO("loading " << view_path << " as sidre view linked to a buffer");
        // we need to fetch the buffer
        int buffer_id = sidre_meta_view["buffer_id"].to_int();

        std::ostringstream buffer_fetch_path_oss;
        buffer_fetch_path_oss << "tree: "
                              << tree_id
                              << " /sidre/buffers/buffer_id_" << buffer_id;

        // buffer data path
        std::string buffer_data_fetch_path   = buffer_fetch_path_oss.str() + "/data";

        // we also need the buffer's schema
        std::string buffer_schema_fetch_path = buffer_fetch_path_oss.str() + "/schema";

        Node n_buffer_schema_str;

        read_from_file_tree(tree_id,
                            buffer_schema_fetch_path,
                            n_buffer_schema_str);

        std::string buffer_schema_str = n_buffer_schema_str.as_string();
        Schema buffer_schema(buffer_schema_str);

        //BP_PLUGIN_INFO("sidre buffer schema: " << buffer_schema.to_json());
        //BP_PLUGIN_INFO("sidre buffer data path " << buffer_data_fetch_path);

        std::string view_schema_str = sidre_meta_view["schema"].as_string();
        // create the schema we want for this view
        // it describes how the view relates to the buffer in the hdf5 file

        Schema view_schema(view_schema_str);
        // BP_PLUGIN_INFO("sidre view schema: " << view_schema.to_json());

        // if the schema isn't compact, or if we are reading 
        // less elements than the entire buffer, 
        // we need to read a subset of the hdf5 dataset

        if(   !view_schema.is_compact() || 
            ( view_schema.dtype().number_of_elements() <
              buffer_schema.dtype().number_of_elements() )
          )
        {
            // BP_PLUGIN_INFO("Sidre View from Buffer Slab Fetch Case");
            //
            // Create a compact schema to describe our desired output data
            //
            Schema view_schema_compact;
            view_schema.compact_to(view_schema_compact);
            // setup and allocate the output node
            out.set(view_schema_compact);

            // ---------------------------------------------------------------
            // BUFFER-SLAB FETCH
            // ---------------------------------------------------------------
            //
            // we can use hdf5 slab fetch if the the dtype.id() of the buffer
            // and the view are the same. 
            // 
            //  otherwise, we will have to fetch the entire buffer since
            //  hdf5 doesn't support byte level striding.

            // if(
            //     tree_cache.Read(tree_id,
            //                     buffer_data_fetch_path,
            //                     view_schema.dtype(),
            //                     out)
            //     )
            // {
                // BP_PLUGIN_INFO("Sidre View from Buffer Slab Fetch Case Failed");
                // ---------------------------------------------------------------
                // Fall back to Non BUFFER-SLAB FETCH
                // ---------------------------------------------------------------
                // this reads the entire buffer to get the proper subset
                Node n_buff;
                Node n_view;

                read_from_file_tree(tree_id,
                                    buffer_data_fetch_path,
                                    n_buff);

                // create our view on the buffer
                n_view.set_external(view_schema,n_buff.data_ptr());
                // compact the view to our output
                n_view.compact_to(out);
            // }
            // else
            // {
            //     //BP_PLUGIN_INFO("Sidre View from Buffer Slab Fetch Case Successful");
            // }
        }
        else // compact, and compat, we can just read
        {
            read_from_file_tree(tree_id,
                                buffer_data_fetch_path,
                                out);
        }
    }
    else if( view_state == "EXTERNAL" )
    {
        //BP_PLUGIN_INFO("loading " << view_path << " as sidre external view");

        std::string fetch_path = "sidre/external/" + view_path;

        // BP_PLUGIN_INFO("relay:io::hdf5_read "
        //                << "domain " << tree_id
        //                << " : "
        //                << fetch_path);

        read_from_file_tree(tree_id,
                            fetch_path,
                            out);
    }
    else
    {
        // error:  "unsupported sidre view state: " << view_state );
    }
}

//-----------------------------------------------------------------------------
void 
SidreHandle::read_from_root(const std::string &path, Node &node)
{
    // skip cache first 
    if(!path.empty())
        m_root_handle.read(path,node);
    else
        m_root_handle.read(node);
}

//-----------------------------------------------------------------------------
void
SidreHandle::prepare_file_handle(int tree_id)
{
    // find if the have the correct io handle open already
    int file_id = generate_file_id_for_tree(tree_id);
    if( m_file_handles.count(file_id) == 0 ||
        !m_file_handles[file_id].is_open() )
    {
        std::cout << " opening: " << generate_file_path(tree_id) << std::endl;
        // if not, open the handle
        m_file_handles[file_id].open(generate_file_path(tree_id));
    }
}

//-----------------------------------------------------------------------------
bool
SidreHandle::file_tree_has_path(int tree_id,
                                const std::string &path)
{
    prepare_file_handle(tree_id);
    int file_id = generate_file_id_for_tree(tree_id);
    // find the proper path for the tree inside the file
    std::string full_path = generate_tree_path(tree_id) + path;
    // check if the file handle has the expected path
    return m_file_handles[file_id].has_path(full_path);
}

//-----------------------------------------------------------------------------
void 
SidreHandle::read_from_file_tree(int tree_id,
                                 const std::string &path,
                                 Node &node)
{
    prepare_file_handle(tree_id);
    int file_id = generate_file_id_for_tree(tree_id);

    // find the proper path for the tree inside the file
    std::string full_path = generate_tree_path(tree_id) + path;

    std::cout << "fetch:" << full_path << std::endl;

    if(full_path.empty() || full_path == "/")
    {
        m_file_handles[file_id].read(node);
    }
    else
    {
        m_file_handles[file_id].read(full_path,node);
    }
}

//-----------------------------------------------------------------------------
void
SidreHandle::read_from_sidre_tree(int tree_id,
                                  const std::string &path,
                                  Node &out)
{
    // if( protocol == "sidre_hdf5" )
    // {

    //Node &sidre_meta = tree_cache.Cache().FetchSidreMetaTree(tree_id);
    Node &sidre_meta = m_sidre_meta[tree_id];

    std::string sidre_mtree_view  = generate_sidre_meta_view_path(path);
    std::string sidre_mtree_group = generate_sidre_meta_group_path(path);

    // this path will either be a sidre group or a sidre view
    // check if either exists cached

    //BP_PLUGIN_INFO("fetch sidre tree: "<< tree_path);

    if( !sidre_meta.has_path(sidre_mtree_group) ||
        !sidre_meta.has_path(sidre_mtree_view) )
    {
        // has_path(tree_id,path)

        // check to see if we have a group or view
        if( file_tree_has_path(tree_id, "sidre/" + sidre_mtree_group) )
        {
            // we have a group, read the meta data
            read_from_file_tree(tree_id, "sidre/" + sidre_mtree_group,
                                sidre_meta[sidre_mtree_group]);

        }
        else if( file_tree_has_path(tree_id, "sidre/" + sidre_mtree_view) )
        {
            // we have a view, read the meta data
            read_from_file_tree(tree_id, "sidre/" + sidre_mtree_view,
                                sidre_meta[sidre_mtree_view]);
        }
        else
        {
            CONDUIT_ERROR("Failed to read tree path: " << std::endl
                          << "Expected to find Sidre Group: "
                          << tree_id << "/sidre/" << sidre_mtree_group
                          << " or "
                          << "Sidre View: "
                          << tree_id << "/sidre/" << sidre_mtree_view);
        }
    }

    load_sidre_tree(sidre_meta,
                    tree_id,
                    path,
                    "",
                    out);

    // // start a top level traversal
    // LoadSidreTree(sidre_meta,
    //               tree_cache,
    //               tree_id,
    //               tree_root,
    //               tree_path,
    //               "",
    //               out);
    // visitTimer->StopTimer(t_sidre_tree, "LoadSidreTree");

    // }
    // else
    // {
    //     // error
    //     // BP_PLUGIN_EXCEPTION1( InvalidVariableException,
    //     //                       "unknown protocol" << protocol);
    // }
}


//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit::relay::<mpi>::io --
//-----------------------------------------------------------------------------

#ifdef CONDUIT_RELAY_IO_MPI_ENABLED
}
//-----------------------------------------------------------------------------
// -- begin conduit::relay::mpi --
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
