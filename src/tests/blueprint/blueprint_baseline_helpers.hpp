// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: blueprint_baseline_helpers.hpp
///
//-----------------------------------------------------------------------------

#ifndef BLUEPRINT_BASELINE_HELPERS_HPP
#define BLUEPRINT_BASELINE_HELPERS_HPP

#include <conduit.hpp>
#include <conduit_node.hpp>
#include <conduit_blueprint.hpp>
#include <conduit_relay.hpp>
#include <conduit_relay_io_blueprint.hpp>

//-----------------------------------------------------------------------------
// -- begin baseline utility functions --
//-----------------------------------------------------------------------------
// NOTE: REQUIRES sep, baseline_dir(), test_name(), get_rank(), barrier()
//  are defined before inclusion.

//-----------------------------------------------------------------------------
#ifdef GENERATE_BASELINES
  #ifdef _WIN32
    #include <direct.h>
    inline void create_path(const std::string &path) { _mkdir(path.c_str()); }
  #else
    #include <sys/stat.h>
    #include <sys/types.h>
    inline void create_path(const std::string &path) { mkdir(path.c_str(), S_IRWXU); }
  #endif
#else
  inline void create_path(const std::string &) {}
#endif

//-----------------------------------------------------------------------------
inline std::string
baseline_file(const std::string &basename)
{
    std::string path(baseline_dir());
    int r = get_rank();
    if(r == 0)
        create_path(path);
    path += (sep + test_name());
    if(r == 0)
        create_path(path);
    path += (sep + basename + ".yaml");
    barrier();
    return path;
}

//-----------------------------------------------------------------------------
inline void
make_baseline(const std::string &filename, const conduit::Node &n)
{
    conduit::relay::io::save(n, filename, "yaml");
}

//-----------------------------------------------------------------------------
inline void
load_baseline(const std::string &filename, conduit::Node &n)
{
    conduit::relay::io::load(filename, "yaml", n);
}

//-----------------------------------------------------------------------------
inline bool
compare_baseline(const std::string &filename, const conduit::Node &n,
    conduit::Node &baseline)
{
    const double tolerance = 1.e-6;
    conduit::Node info;
    conduit::relay::io::load(filename, "yaml", baseline);

    // Node::diff returns true if the nodes are different. We want not different.
    bool equal = !baseline.diff(n, info, tolerance, true);

    if(!equal)
    {
        const char *line = "*************************************************************";
        std::cout << "Difference!" << std::endl;
        std::cout << line << std::endl;

        conduit::Node opts;
        opts["num_elements_threshold"] = 20;
        opts["num_children_threshold"] = 10000;
        info.to_summary_string_stream(std::cout, opts);
    }
    return equal;
}

//-----------------------------------------------------------------------------
inline bool
compare_baseline(const std::string &filename, const conduit::Node &n)
{
    conduit::Node baseline;
    return compare_baseline(filename, n, baseline);
}

//-----------------------------------------------------------------------------
inline bool
check_if_hdf5_enabled()
{
    conduit::Node io_protos;
    conduit::relay::io::about(io_protos["io"]);
    return io_protos["io/protocols/hdf5"].as_string() == "enabled";
}

//-----------------------------------------------------------------------------
inline void
save_node(const std::string &filename, const conduit::Node &mesh)
{
    conduit::relay::io::blueprint::save_mesh(mesh, filename + ".yaml", "yaml");
}

//-----------------------------------------------------------------------------
inline void
save_visit(const std::string &filename, const conduit::Node &n, bool use_subdir = false)
{
#ifdef GENERATE_BASELINES
    // NOTE: My VisIt only wants to read HDF5 root files for some reason.
    bool hdf5_enabled = check_if_hdf5_enabled();

    auto pos = filename.rfind("/");
    std::string fn(filename.substr(pos+1,filename.size()-pos-1));
    pos = fn.rfind(".");
    std::string fn_noext(fn.substr(0, pos));


    // Save all the domains to individual files.
    auto ndoms = conduit::blueprint::mesh::number_of_domains(n);
    if(ndoms < 1)
        return;

    // All domains go into subdirectory
    if(!conduit::utils::is_directory(fn_noext) && use_subdir)
    {
        conduit::utils::create_directory(fn_noext);
    }

    char dnum[20];
    if(ndoms == 1)
    {
        sprintf(dnum, "%05d", 0);
        std::stringstream ss;
        if(use_subdir)
        {
            ss << fn_noext << conduit::utils::file_path_separator()
                << fn_noext << "." << dnum;
        }
        else
        {
            ss << fn_noext << "." << dnum;
        }

        if(hdf5_enabled)
            conduit::relay::io::save(n, ss.str() + ".hdf5", "hdf5");
        // VisIt won't read it:
        conduit::relay::io::save(n, ss.str() + ".yaml", "yaml");
    }
    else
    {
        for(size_t i = 0; i < ndoms; i++)
        {
            sprintf(dnum, "%05d", static_cast<int>(i));
            std::stringstream ss;
            if(use_subdir)
            {
                ss << fn_noext << conduit::utils::file_path_separator()
                    << fn_noext << "." << dnum;
            }
            else
            {
                ss << fn_noext << "." << dnum;
            }

            if(hdf5_enabled)
                conduit::relay::io::save(n[i], ss.str() + ".hdf5", "hdf5");
            // VisIt won't read it:
            conduit::relay::io::save(n[i], ss.str() + ".yaml", "yaml");
        }
    }

    // Add index stuff to it so we can plot it in VisIt.
    conduit::Node root;
    if(ndoms == 1)
        conduit::blueprint::mesh::generate_index(n, "", ndoms, root["blueprint_index/mesh"]);
    else
        conduit::blueprint::mesh::generate_index(n[0], "", ndoms, root["blueprint_index/mesh"]);
    root["protocol/name"] = "hdf5";
    root["protocol/version"] = CONDUIT_VERSION;
    root["number_of_files"] = ndoms;
    root["number_of_trees"] = ndoms;
    if(use_subdir)
    {
        root["file_pattern"] = fn_noext + conduit::utils::file_path_separator() + (fn_noext + ".%05d.hdf5");
    }
    else
    {
        root["file_pattern"] = (fn_noext + ".%05d.hdf5");
    }
    root["tree_pattern"] = "/";

    if(hdf5_enabled)
        conduit::relay::io::save(root, fn_noext + "_hdf5.root", "hdf5");

    if(use_subdir)
    {
       root["file_pattern"] = fn_noext + conduit::utils::file_path_separator() + (fn_noext + ".%05d.yaml");
    }
    else
    {
        root["file_pattern"] = (fn_noext + ".%05d.yaml");
    }
    // VisIt won't read it:
    conduit::relay::io::save(root, fn_noext + "_yaml.root", "yaml");
#endif
}

//-----------------------------------------------------------------------------
// -- end baseline utility functions --
//-----------------------------------------------------------------------------

#endif
