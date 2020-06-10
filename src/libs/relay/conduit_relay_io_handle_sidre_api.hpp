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
/// file: conduit_relay_io_handle_sidre.inc.cpp
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// SidreHandle -- IO Handle implementation for Sidre Style Files
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
class SidreHandle: public IOHandle::HandleInterface
{
public:
    SidreHandle(const std::string &path,
               const std::string &protocol,
               const Node &options);
    virtual ~SidreHandle();

    void open();

    bool is_open() const;

    // main interface methods
    void read(Node &node);
    void read(const std::string &path,
              Node &node);

    void write(const Node &node);
    void write(const Node &node,
               const std::string &path);

    void remove(const std::string &path);

    void list_child_names(std::vector<std::string> &res) const;
    void list_child_names(const std::string &path,
                          std::vector<std::string> &res) const;

    bool has_path(const std::string &path) const;

    void close();


private:

    std::string root_file_directory() const;
    std::string detect_root_protocol() const;

    std::string expand_pattern(const std::string pattern,
                               int idx) const;

    int         generate_file_id_for_tree(int tree_id) const;
    std::string generate_file_path(int tree_id) const;
    std::string generate_tree_path(int tree_id) const;
    void        generate_domain_to_file_map(int num_domains,
                                            int num_files,
                                            Node &out) const;
    std::string generate_sidre_meta_group_path(const std::string &tree_path) const;
    std::string generate_sidre_meta_view_path(const std::string &tree_path)  const;


    void load_sidre_tree(Node &sidre_meta,
                         int tree_id,
                         const std::string &tree_path,
                         const std::string &curr_path,
                         Node &out);

    void load_sidre_group(Node &sidre_meta,
                          int tree_id,
                          const std::string &group_path,
                          Node &out);

    void load_sidre_view(Node &sidre_meta_view,
                         int tree_id,
                         const std::string &view_path,
                         Node &out);


    void read_from_root(const std::string &path,
                        Node &node);

    void read_from_sidre_tree(int tree_id,
                              const std::string &path,
                              Node &node);

    void prepare_file_handle(int tree_id);

    bool file_tree_has_path(int tree_id,
                            const std::string &path);


    void read_from_file_tree(int tree_id,
                             const std::string &path,
                             Node &node);

    bool                     m_open;
    int                      m_num_trees;
    int                      m_num_files;

    std::string              m_root_protocol;
    std::string              m_root_file;

    std::string              m_file_pattern;
    std::string              m_tree_pattern;
    std::string              m_file_protocol;

    // io handle used to interacte with the sidre root file
    IOHandle                 m_root_handle;

    // holds open I/O handles for each tree
    std::map<int,IOHandle>   m_file_handles;
    // holds cached sidre meta date for each tree
    std::map<int,Node>       m_sidre_meta;
    
};
