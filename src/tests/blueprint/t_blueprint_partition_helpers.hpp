// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_blueprint_partition_helpers.hpp
///
//-----------------------------------------------------------------------------

#ifndef T_BLUEPRINT_PARTITION_HELPERS_HPP
#define T_BLUEPRINT_PARTITION_HELPERS_HPP

//-----------------------------------------------------------------------------
#ifdef GENERATE_BASELINES
  #ifdef _WIN32
    #include <direct.h>
    void create_path(const std::string &path) { _mkdir(path.c_str()); }
  #else
    #include <sys/stat.h>
    #include <sys/types.h>
    void create_path(const std::string &path) { mkdir(path.c_str(), S_IRWXU); }
  #endif
#else
  void create_path(const std::string &) {}
#endif

//-----------------------------------------------------------------------------
std::string
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
void
make_baseline(const std::string &filename, const conduit::Node &n)
{
    conduit::relay::io::save(n, filename, "yaml");
}

//-----------------------------------------------------------------------------
void
load_baseline(const std::string &filename, conduit::Node &n)
{
    conduit::relay::io::load(filename, "yaml", n);
}

//-----------------------------------------------------------------------------
bool
compare_baseline(const std::string &filename, const conduit::Node &n)
{
    const double tolerance = 1.e-6;
    conduit::Node baseline, info;
    conduit::relay::io::load(filename, "yaml", baseline);

    // Node::diff returns true if the nodes are different. We want not different.
    bool equal = !baseline.diff(n, info, tolerance, true);

    if(!equal)
    {
       const char *line = "*************************************************************";
       std::cout << "Difference!" << std::endl;
       std::cout << line << std::endl;
       info.print();
    }
    return equal;
}

//-----------------------------------------------------------------------------
bool
check_if_hdf5_enabled()
{
    conduit::Node io_protos;
    conduit::relay::io::about(io_protos["io"]);
    return io_protos["io/protocols/hdf5"].as_string() == "enabled";
}

//-----------------------------------------------------------------------------
void
save_node(const std::string &filename, const conduit::Node &mesh)
{
    conduit::relay::io::blueprint::save_mesh(mesh, filename + ".yaml", "yaml");
}

//-----------------------------------------------------------------------------
void
save_visit(const std::string &filename, const conduit::Node &n)
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
    char dnum[20];
    if(ndoms == 1)
    {
        sprintf(dnum, "%05d", 0);
        std::stringstream ss;
        ss << fn_noext << "." << dnum;

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
            ss << fn_noext << "." << dnum;

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
    root["file_pattern"] = (fn_noext + ".%05d.hdf5");
    root["tree_pattern"] = "/";

    if(hdf5_enabled)
        conduit::relay::io::save(root, fn_noext + "_hdf5.root", "hdf5");

    root["file_pattern"] = (fn_noext + ".%05d.yaml");
    // VisIt won't read it:
    conduit::relay::io::save(root, fn_noext + "_yaml.root", "yaml");
#endif
}

//-----------------------------------------------------------------------------
/**
Make a field that selects domains like this.
+----+----+
| 3  |  5 |
|  +-|-+  |
|  +4|4|  |
+--+-+-+--|
|  +1|1|  |
|  +-|-+  |
| 0  |  2 |
+----+----+

*/
void
add_field_selection_field(int cx, int cy, int cz,
    int iquad, int jquad, conduit::index_t main_dom, conduit::index_t fill_dom,
    conduit::Node &output)
{
    std::vector<conduit::int64> values(cx*cy*cz, main_dom);
    int sq = 2*jquad + iquad;
    int idx = 0;
    for(int k = 0; k < cz; k++)
    for(int j = 0; j < cy; j++)
    for(int i = 0; i < cx; i++)
    {
        int ci = (i < cx/2) ? 0 : 1;
        int cj = (j < cy/2) ? 0 : 1;
        int csq = 2*cj + ci;
        if(csq == sq)
            values[idx] = fill_dom;
        idx++;
    }
    output["fields/selection_field/type"] = "scalar";
    output["fields/selection_field/association"] = "element";
    output["fields/selection_field/topology"] = "mesh";
    output["fields/selection_field/values"].set(values);
}

void
make_field_selection_example(conduit::Node &output, int mask)
{
    int nx = 11, ny = 11, nz = 3;
    int m = 1, dc = 0;
    for(int i = 0; i < 4; i++)
    {
        if(m & mask)
            dc++;
        m <<= 1;
    }

    if(mask & 1)
    {
        conduit::Node &dom0 = (dc > 1) ? output.append() : output;
        conduit::blueprint::mesh::examples::braid("uniform", nx, ny, nz, dom0);
        dom0["state/cycle"] = 1;
        dom0["state/domain_id"] = 0;
        dom0["coordsets/coords/origin/x"] = 0.;
        dom0["coordsets/coords/origin/y"] = 0.;
        dom0["coordsets/coords/origin/z"] = 0.;
        add_field_selection_field(nx-1, ny-1, nz-1, 1,1, 0, 11, dom0);
    }

    if(mask & 2)
    {
        conduit::Node &dom1 = (dc > 1) ? output.append() : output;
        conduit::blueprint::mesh::examples::braid("uniform", nx, ny, nz, dom1);
        auto dx = dom1["coordsets/coords/spacing/dx"].to_float();
        dom1["state/cycle"] = 1;
        dom1["state/domain_id"] = 1;
        dom1["coordsets/coords/origin/x"] = dx * static_cast<double>(nx-1);
        dom1["coordsets/coords/origin/y"] = 0.;
        dom1["coordsets/coords/origin/z"] = 0.;
        add_field_selection_field(nx-1, ny-1, nz-1, 0,1, 22, 11, dom1);
    }

    if(mask & 4)
    {
        conduit::Node &dom2 = (dc > 1) ? output.append() : output;
        conduit::blueprint::mesh::examples::braid("uniform", nx, ny, nz, dom2);
        auto dy = dom2["coordsets/coords/spacing/dy"].to_float();
        dom2["state/cycle"] = 1;
        dom2["state/domain_id"] = 2;
        dom2["coordsets/coords/origin/x"] = 0.;
        dom2["coordsets/coords/origin/y"] = dy * static_cast<double>(ny-1);
        dom2["coordsets/coords/origin/z"] = 0.;
        add_field_selection_field(nx-1, ny-1, nz-1, 1,0, 33, 44, dom2);
    }

    if(mask & 8)
    {
        conduit::Node &dom3 = (dc > 1) ? output.append() : output;
        conduit::blueprint::mesh::examples::braid("uniform", nx, ny, nz, dom3);
        auto dx = dom3["coordsets/coords/spacing/dx"].to_float();
        auto dy = dom3["coordsets/coords/spacing/dy"].to_float();
        dom3["state/cycle"] = 1;
        dom3["state/domain_id"] = 3;
        dom3["coordsets/coords/origin/x"] = dx * static_cast<double>(nx-1);
        dom3["coordsets/coords/origin/y"] = dy * static_cast<double>(ny-1);
        dom3["coordsets/coords/origin/z"] = 0.;
        add_field_selection_field(nx-1, ny-1, nz-1, 0,0, 55, 44, dom3);
    }
}

#endif
