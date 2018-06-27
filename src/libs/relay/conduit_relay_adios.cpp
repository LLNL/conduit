//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2014-2018, Lawrence Livermore National Security, LLC.
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
/// file: conduit_relay_adios.cpp
///
//-----------------------------------------------------------------------------

#ifdef USE_MPI
#include "conduit_relay_mpi_io_adios.hpp"
#else
  // Force serial ADIOS using _NOMPI
  #ifndef _NOMPI
  #define _NOMPI
  #endif
#include "conduit_relay_io_adios.hpp"
#endif

//-----------------------------------------------------------------------------
// standard lib includes
//-----------------------------------------------------------------------------
#include <iostream>
#include <algorithm>
#include <cstdio>
#include <cstring>
#ifdef MAKE_SEPARATE_GROUPS
#include <sys/time.h>
#endif

//-----------------------------------------------------------------------------
// external lib includes
//-----------------------------------------------------------------------------
#include <adios.h>
#include <adios_read.h>
#include <adios_transform_methods.h>

//-----------------------------------------------------------------------------
// -- conduit includes -- 
//-----------------------------------------------------------------------------
#include "conduit_error.hpp"

using std::cout;
using std::endl;

#define CURRENT_TIME_STEP -1

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

#ifdef USE_MPI
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
// -- begin conduit::relay::<mpi>::io::internals --
//-----------------------------------------------------------------------------
namespace internals
{

static int rank = 0;
static int size = 1;

//#define DEBUG_PRINT_RANK(OSSEXPR) if(internals::rank == 0) { cout << OSSEXPR << endl; }
#define DEBUG_PRINT_RANK(OSSEXPR) 

//-----------------------------------------------------------------------------
void print_varinfo(std::ostream &os, ADIOS_FILE *afile, ADIOS_VARINFO *v)
{
    os << "vname=" << afile->var_namelist[v->varid] << endl << "{" << endl;
    os << "   type = " << v->type << endl;
    os << "   ndim = " << v->ndim << endl;
    os << "   dims = {";
    for(int q = 0; q < v->ndim; ++q)
        os << "," << v->dims[q];
    os << "}" << endl;
    os << "   nsteps = " << v->nsteps << endl;
    os << "   global = " << v->global << endl;
    os << "   nblocks = {";
    for(int q = 0; q < v->nsteps; ++q)
        os << "," << v->nblocks[q];
    os << "}" << endl;
    os << "   sum_nblocks = " << v->sum_nblocks << endl;
    os << "   nattrs = " << v->nattrs << endl;
    os << "   attr_ids = {";
    for(int q = 0; q < v->nattrs; ++q)
    {
        os << "," << v->attr_ids[q] << " (" << afile->attr_namelist[v->attr_ids[q]];
        if(strstr(afile->attr_namelist[v->attr_ids[q]], "conduit_type") != NULL)
        {
            std::string avalue;
            ADIOS_DATATYPES atype;
            int asize = 0;
            void *aptr = NULL;
            adios_get_attr_byid(afile, v->attr_ids[q], &atype, &asize, &aptr);
            if(atype == adios_string)
            {
                avalue = std::string((const char *)aptr);
            }
            free(aptr);

            os << " = " << avalue;
        }
        os << ")";
    }
    os << "}" << endl;

    os << "   blockinfo = {" << endl;
    for(int bi = 0; bi < v->sum_nblocks; ++bi)
    {
        os << "       {" << endl;

        os << "           start={";
        for(int q = 0; q < v->ndim; ++q)
            os << "," << v->blockinfo[bi].start[q];
        os << "}" << endl;

        os << "           count={";
        for(int q = 0; q < v->ndim; ++q)
            os << "," << v->blockinfo[bi].count[q];
        os << "}" << endl;

        os << "           process_id=" << v->blockinfo[bi].process_id << endl;
        os << "           time_index=" << v->blockinfo[bi].time_index << endl;
        os << "       }," << endl;
    }
    os << "   }" << endl;

    os << "}" << endl;
}

//-----------------------------------------------------------------------------
std::string
read_method_to_string(ADIOS_READ_METHOD m)
{
    std::string s("ADIOS_READ_METHOD_BP");
    if(m == ADIOS_READ_METHOD_BP)
        s = "ADIOS_READ_METHOD_BP";
    else if(m == ADIOS_READ_METHOD_BP_AGGREGATE)
        s = "ADIOS_READ_METHOD_BP_AGGREGATE";
    else if(m == ADIOS_READ_METHOD_DATASPACES)
        s = "ADIOS_READ_METHOD_DATASPACES";
    else if(m == ADIOS_READ_METHOD_DIMES)
        s = "ADIOS_READ_METHOD_DIMES";
    else if(m == ADIOS_READ_METHOD_FLEXPATH)
        s = "ADIOS_READ_METHOD_FLEXPATH";
    else if(m == ADIOS_READ_METHOD_ICEE)
        s = "ADIOS_READ_METHOD_ICEE";
    return s;
}

//-----------------------------------------------------------------------------
ADIOS_READ_METHOD
string_to_read_method(const std::string &s)
{
    ADIOS_READ_METHOD m = ADIOS_READ_METHOD_BP;
    if(s == "ADIOS_READ_METHOD_BP")
        m = ADIOS_READ_METHOD_BP;
    else if(s == "ADIOS_READ_METHOD_BP_AGGREGATE")
        m = ADIOS_READ_METHOD_BP_AGGREGATE;
    else if(s == "ADIOS_READ_METHOD_DATASPACES")
        m = ADIOS_READ_METHOD_DATASPACES;
    else if(s == "ADIOS_READ_METHOD_DIMES")
        m = ADIOS_READ_METHOD_DIMES;
    else if(s == "ADIOS_READ_METHOD_FLEXPATH")
        m = ADIOS_READ_METHOD_FLEXPATH;
    else if(s == "ADIOS_READ_METHOD_ICEE")
        m = ADIOS_READ_METHOD_ICEE;
    return m;
}

//-----------------------------------------------------------------------------
std::string
lock_mode_to_string(ADIOS_LOCKMODE m)
{
    std::string s("ADIOS_LOCKMODE_NONE");
    if(m == ADIOS_LOCKMODE_NONE)
        s = "ADIOS_LOCKMODE_NONE";
    else if(m == ADIOS_LOCKMODE_CURRENT)
        s = "ADIOS_LOCKMODE_CURRENT";
    else if(m == ADIOS_LOCKMODE_ALL)
        s = "ADIOS_LOCKMODE_ALL";
    return s;
}

//-----------------------------------------------------------------------------
ADIOS_LOCKMODE
string_to_lock_mode(const std::string &s)
{
    ADIOS_LOCKMODE m = ADIOS_LOCKMODE_NONE;
    if(s == "ADIOS_LOCKMODE_NONE")
        m = ADIOS_LOCKMODE_NONE;
    else if(s == "ADIOS_LOCKMODE_CURRENT")
        m = ADIOS_LOCKMODE_CURRENT;
    else if(s == "ADIOS_LOCKMODE_ALL")
        m = ADIOS_LOCKMODE_ALL;
    return m;
}

//-----------------------------------------------------------------------------
std::string
statistics_flag_to_string(ADIOS_STATISTICS_FLAG f)
{
    std::string s("adios_stat_no");
    if(f == adios_stat_default)
        s = "adios_stat_default";
    else if(f == adios_stat_no)
        s = "adios_stat_no";
    else if(f == adios_stat_minmax)
        s = "adios_stat_minmax";
    else if(f == adios_stat_full)
        s = "adios_stat_full";
    return s;
}

//-----------------------------------------------------------------------------
ADIOS_STATISTICS_FLAG
string_to_statistics_flag(const std::string &s)
{
    ADIOS_STATISTICS_FLAG f = adios_stat_no;
    if(s == "adios_stat_default")
        f = adios_stat_default;
    else if(s == "adios_stat_no")
        f = adios_stat_no;
    else if(s == "adios_stat_minmax")
        f = adios_stat_minmax;
    else if(s == "adios_stat_full")
        f = adios_stat_full;
    return f;
}

//-----------------------------------------------------------------------------
bool
streamIsFileBased(ADIOS_READ_METHOD method)
{
    switch(method)
    {
    case ADIOS_READ_METHOD_BP:
    case ADIOS_READ_METHOD_BP_AGGREGATE:
        return true;
    case ADIOS_READ_METHOD_DATASPACES:
    case ADIOS_READ_METHOD_DIMES:
    case ADIOS_READ_METHOD_FLEXPATH:
    case ADIOS_READ_METHOD_ICEE:
        return false;
    }
    return false;
}

//-----------------------------------------------------------------------------
#ifdef MAKE_SEPARATE_GROUPS
static unsigned long
current_time_stamp()
{
    // Get the current time of day.
    timeval t;
    gettimeofday(&t, 0);
    // Convert to microseconds since epoch.
    return t.tv_sec * 1000000 + t.tv_usec;
}
#endif

//-----------------------------------------------------------------------------
static void
create_group_name(char name[32])
{
#ifdef MAKE_SEPARATE_GROUPS
    unsigned long t = current_time_stamp();
    sprintf(name, "conduit%020ld", t);
#else
    strcpy(name, "conduit");
#endif
}

//-----------------------------------------------------------------------------
static std::vector<std::string> available_transports()
{
    std::vector<std::string> v;
    ADIOS_AVAILABLE_WRITE_METHODS *wm = adios_available_write_methods();
    if(wm)
    {
        for (int i=0; i<wm->nmethods; i++)
            v.push_back(wm->name[i]);
        adios_available_write_methods_free(wm); 
    }
    return v;
}

//-----------------------------------------------------------------------------
static std::vector<std::string> available_transforms()
{
    std::vector<std::string> v;
#if 1
    // Workaround
    v.push_back("identity");
    v.push_back("zfp");
#else
    // Crashing
    ADIOS_AVAILABLE_TRANSFORM_METHODS *t = adios_available_transform_methods();
    if(t)
    {
        for (int i=0; i<t->ntransforms; i++)
            v.push_back(t->name[i]);
        adios_available_transform_methods_free(t); 
    }
#endif
    return v;
}

//-----------------------------------------------------------------------------
static bool
string_vector_contains(const std::vector<std::string> &sv, const std::string &s)
{
    return std::find(sv.begin(), sv.end(), s) != sv.end();
}

//-----------------------------------------------------------------------------
bool is_integer(const std::string &s, int &ivalue)
{
    return sscanf(s.c_str(), "%d", &ivalue) == 1;
}

//-----------------------------------------------------------------------------
bool is_positive_integer(const std::string &s, int &ivalue)
{
    bool ret = is_integer(s, ivalue);
    return ret && ivalue >= 0;
}

//-----------------------------------------------------------------------------
// NOTE: Move to conduit::utils?
void
splitpath(const std::string &path, std::string &filename,
    int &time_step, int &domain, std::vector<std::string> &subpaths, 
    bool prefer_time = false)
{
    std::vector<std::string> tok;
    conduit::utils::split_string(path, ':', tok);

    if(tok.empty())
        filename = path; // Would have had to be an empty string.
    else if(tok.size() == 1)
        filename = path;
    else if(tok.size() == 2)
    {
        filename = tok[0];
        int ivalue = 0;
        if(is_integer(tok[1], ivalue))
        {
            if(prefer_time)
            {
                // filename:timestep
                time_step = ivalue;
            }
            else
            {
                // filename:domain
                domain = ivalue;
            }
        }
        else
        {
            // filename:subpaths
            subpaths.push_back(tok[1]);
        }
    }
    else if(tok.size() >= 3)
    {
        filename = tok[0];
        int ivalue1 = 0, ivalue2 = 0;
        bool arg1 = is_integer(tok[1], ivalue1);
        bool arg2 = is_positive_integer(tok[2], ivalue2);
        if(arg1 && arg2)
        {
            // filename:timestep:domain
            time_step = ivalue1;
            domain    = ivalue2;
        }
        else if(arg1 && !arg2)
        {
            // filename:domain:subpaths
            domain = ivalue1;
            subpaths.push_back(tok[2]);
        }
        else if(!arg1 && arg2)
        {
            // filename:<non-int>:int
            // Assume these are just numeric subpaths for now.
            // We could test for tok[1] == "current" to denote time...
            subpaths.push_back(tok[1]);
            subpaths.push_back(tok[2]);
        }
        else // !arg1 && !arg2
        {
            // filename:subpath:subpath
            subpaths.push_back(tok[1]);
            subpaths.push_back(tok[2]);
        }

        // Save the remaining tokens as subpaths
        for(size_t i = 3; i < tok.size(); ++i)
            subpaths.push_back(tok[i]);
    }
}

//-----------------------------------------------------------------------------
// Private class used to hold options that control adios i/o params.
// 
// These values are read by about(), and are set by io::adios_set_options()
// 
// NOTE: the options are split into separate nodes in the Conduit about/set
//       so the options will be more obvious to the querying application. The
//       alternative is to make them all just a string and let the user have
//       to guess about what the options are.
//
//-----------------------------------------------------------------------------

class ADIOSOptions
{
public:
    long                  buffer_size;
    std::string           transport;
    ADIOS_STATISTICS_FLAG statistics_flag;
    std::string           transform;
    std::string transport_options;
    std::string transform_options;

    ADIOS_READ_METHOD read_method;
    std::string       read_parameters;
    ADIOS_LOCKMODE    read_lock_mode;
    int               read_verbose;
    float             read_timeout;
public:
    ADIOSOptions() :
        // Write options
        buffer_size(1024*1024), transport("POSIX"), 
        statistics_flag(adios_stat_no), transform(),
        transport_options(),
        transform_options(),

        // Read options
        read_method(ADIOS_READ_METHOD_BP),
        read_parameters(),
        read_lock_mode(ADIOS_LOCKMODE_ALL), //ADIOS_LOCKMODE_CURRENT would be for streams.
        read_verbose(0),
        read_timeout(-1.f) // block by default
    {
#ifdef USE_MPI
        transport = "MPI";
#endif
    }
    
    //------------------------------------------------------------------------
    void set(const Node &opts)
    {
        // Write options
        if(opts.has_child("write"))
        {
            const Node &n = opts["write"];

            if(n.has_child("buffer_size"))
                buffer_size = n["buffer_size"].as_long();

            if(n.has_child("transport"))
            {
                std::string s(n["transport"].as_string());
                if(string_vector_contains(available_transports(), s))
                    transport = s;
            }

            if(n.has_child("statistics_flag"))
            {
                std::string s(n["statistics_flag"].as_string());
                statistics_flag = internals::string_to_statistics_flag(s);
            }

            if(n.has_child("transform"))
            {
                std::string s(n["transform"].as_string());
                if(s == "" || 
                   string_vector_contains(available_transforms(), s))
                {
                    transform = s;
                }
            }

            if(n.has_child("transport_options"))
                transport_options = n["transport_options"].as_string();

            if(n.has_child("transform_options"))
                transform_options = n["transform_options"].as_string();
        }

        // Read options
        if(opts.has_child("read"))
        {
            const Node &n = opts["read"];
            if(n.has_child("read_method"))
            {
                std::string s(n["read_method"].as_string());
                read_method = internals::string_to_read_method(s);
            }

            if(n.has_child("parameters"))
                read_parameters = n["parameters"].as_string();

            if(n.has_child("lock_mode"))
            {
                std::string s(n["lock_mode"].as_string());
                read_lock_mode = internals::string_to_lock_mode(s);
            }

            if(n.has_child("timeout"))
                read_timeout = n["timeout"].to_value();

            if(n.has_child("verbose"))
                read_verbose = n["verbose"].to_value();
        }
    }

    //------------------------------------------------------------------------
    void about(Node &opts)
    {
        opts.reset();

        // Write options.
        opts["write/buffer_size"] = buffer_size;
        opts["write/transport"] = transport;
        opts["write/transport_options"] = transport_options;
        opts["write/transform"] = transform;
        opts["write/transform_options"] = transform_options;
        opts["write/statistics_flag"] = 
            internals::statistics_flag_to_string(statistics_flag);

        //
        // Add some read options.
        //
        if(read_method == ADIOS_READ_METHOD_BP)
            opts["read/method"] = "ADIOS_READ_METHOD_BP";
        else if(read_method == ADIOS_READ_METHOD_BP_AGGREGATE)
            opts["read/method"] = "ADIOS_READ_METHOD_BP_AGGREGATE";
        else if(read_method == ADIOS_READ_METHOD_DATASPACES)
            opts["read/method"] = "ADIOS_READ_METHOD_DATASPACES";
        else if(read_method == ADIOS_READ_METHOD_DIMES)
            opts["read/method"] = "ADIOS_READ_METHOD_DIMES";
        else if(read_method == ADIOS_READ_METHOD_FLEXPATH)
            opts["read/method"] = "ADIOS_READ_METHOD_FLEXPATH";

        opts["read/parameters"] = read_parameters;

        if(read_lock_mode == ADIOS_LOCKMODE_NONE)
            opts["read/lock_mode"] = "";
        else if(read_lock_mode == ADIOS_LOCKMODE_CURRENT)
            opts["read/lock_mode"] = "ADIOS_LOCKMODE_CURRENT";
        else if(read_lock_mode == ADIOS_LOCKMODE_ALL)
            opts["read/lock_mode"] = "ADIOS_LOCKMODE_ALL";

        opts["read/verbose"] = read_verbose;
        opts["read/timeout"] = read_timeout;

        // Add in the available transports and transforms.
        std::string sep(", ");
        std::vector<std::string> transports(available_transports()),
                                 transforms(available_transforms());
        std::string key_transport("information/available_transports");
        opts[key_transport].set(DataType::list());
        for(size_t i = 0; i < transports.size(); ++i)
            opts[key_transport].append().set(transports[i]);
        std::string key_transform("information/available_transforms");
        opts[key_transform].set(DataType::list());
        for(size_t i = 0; i < transforms.size(); ++i)
            opts[key_transform].append().set(transforms[i]);
    }
};

// default adios i/o settings
static ADIOSOptions *adiosState_options = NULL;

//-----------------------------------------------------------------------------
// @brief Clean up the options at exit.
static void cleanup_options(void)
{
    if(adiosState_options != NULL)
    {
        delete adiosState_options;
        adiosState_options = NULL;
    }
}

//-----------------------------------------------------------------------------
// @brief Access the ADIOS save options, creating them first if needed. 
//        We create them on the heap to make sure that the object does
//        not fail to initialize statically.
static ADIOSOptions *options()
{
    if(adiosState_options == NULL)
    {
        adiosState_options = new ADIOSOptions;
        atexit(cleanup_options);
    }
    return adiosState_options;
}

//-----------------------------------------------------------------------------
static bool adios_initialized = false;
#define MAX_READ_METHODS 15
static int read_methods_initialized[MAX_READ_METHODS];

static void initialize(
    CONDUIT_RELAY_COMMUNICATOR_ARG0(MPI_Comm comm)
    )
{
    //cout << "initialize: init'd=" << adios_initialized << endl;
    if(!adios_initialized)
    {
        // Mark that we have not initialized any read methods. Read methods
        // are not initialized until we use them.
        memset(read_methods_initialized, 0, sizeof(int)*MAX_READ_METHODS);

#ifdef USE_MPI
        // See if MPI is initialized.
        //int mi = 0;
        //MPI_Initialized(&mpi_init);
        //cout << "initializing parallel ADIOS: mpi_init = " << mi << endl;

        // Initialize ADIOS.
        adios_init_noxml(comm);
#else
        //cout << "initializing serial ADIOS." << endl;
        // Initialize ADIOS.
        adios_init_noxml(0);
#endif
        adios_initialized = true;
    }
}

//-----------------------------------------------------------------------------
static void mark_read_method_initialized(ADIOS_READ_METHOD m)
{
    read_methods_initialized[static_cast<int>(m)] = 1;
}

//-----------------------------------------------------------------------------
static void finalize_read_methods()
{
    for(int i = 0; i < MAX_READ_METHODS; ++i)
    {
        if(read_methods_initialized[i])
        {
            // cout << "adios_read_finalize_method("
            //      << static_cast<ADIOS_READ_METHOD>(i)<< ")" << endl;
            adios_read_finalize_method(static_cast<ADIOS_READ_METHOD>(i));
            read_methods_initialized[i] = 0;
        }
    }
}

//-----------------------------------------------------------------------------
static void finalize(
    CONDUIT_RELAY_COMMUNICATOR_ARG0(MPI_Comm comm)
    )
{
    cleanup_options();
    finalize_read_methods();
    // cout << "adios_finalize()" << endl;
#ifdef USE_MPI
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    adios_finalize(rank);
#else
    adios_finalize(0);
#endif
}

//-----------------------------------------------------------------------------
static void iterate_conduit_node(const conduit::Node &node,
    void (*func)(const Node &,
                 void *,
#ifdef USE_MPI
                 MPI_Comm
#else
                 int
#endif
                ),
    void *funcData,
#ifdef USE_MPI
    MPI_Comm comm
#else
    int comm
#endif
    )
{
    if(node.number_of_children() == 0)
    {
        func(node, funcData, comm);
    }
    else
    {
        for(conduit::index_t i = 0; i < node.number_of_children(); ++i)
            iterate_conduit_node(node.child(i), func, funcData, comm);
    }  
}

//-----------------------------------------------------------------------------
struct adios_save_state
{
    adios_save_state() : fid(0), gid(0), gSize(0), adios_path()
    {
        memset(groupName, 0, 32 * sizeof(char));
    }

    int64_t  fid;
    int64_t  gid;
    uint64_t gSize;
    char        groupName[32];
    std::string adios_path;
};

//-----------------------------------------------------------------------------
static ADIOS_DATATYPES conduit_dtype_to_adios_dtype(const Node &node)
{
    ADIOS_DATATYPES dtype = adios_unknown;
    if(!node.dtype().is_empty() &&
       !node.dtype().is_object() &&
       !node.dtype().is_list())
    {
        if(node.dtype().is_number())
        {
            if(node.dtype().is_int8())
                dtype = adios_byte;
            else if(node.dtype().is_int16())
                dtype = adios_short;
            else if(node.dtype().is_int32())
                dtype = adios_integer;
            else if(node.dtype().is_int64())
                dtype = adios_long;
            else if(node.dtype().is_uint8())
                dtype = adios_unsigned_byte;
            else if(node.dtype().is_uint16())
                dtype = adios_unsigned_short;
            else if(node.dtype().is_uint32())
                dtype = adios_unsigned_integer;
            else if(node.dtype().is_uint64())
                dtype = adios_unsigned_long;
            else if(node.dtype().is_float32())
                dtype = adios_real;
            else if(node.dtype().is_float64())
                dtype = adios_double;
//            else if(node.dtype().is_index_t()) // Conduit does not implement
//                dtype = adios_unsigned_long; // ???
            else if(node.dtype().is_char())
                dtype = adios_byte;
            else if(node.dtype().is_short())
                dtype = adios_short;
            else if(node.dtype().is_int())
                dtype = adios_integer;
            else if(node.dtype().is_long())
                dtype = adios_long;
            else if(node.dtype().is_unsigned_char())
                dtype = adios_unsigned_byte;
            else if(node.dtype().is_unsigned_short())
                dtype = adios_unsigned_short;
            else if(node.dtype().is_unsigned_int())
                dtype = adios_unsigned_integer;
            else if(node.dtype().is_unsigned_long())
                dtype = adios_unsigned_long;
            else if(node.dtype().is_float())
                dtype = adios_real;
            else if(node.dtype().is_double())
                dtype = adios_double;
        }
        else if(node.dtype().is_string() || 
                node.dtype().is_char8_str())
        {
            // NOTE: store as byte arrays since adios_string is really
            //       just for attributes and some scalars.
            dtype = adios_byte;
        }
    }

    return dtype;
}

//-----------------------------------------------------------------------------
static void define_variables(const Node &node, void *funcData,
#ifdef USE_MPI
    MPI_Comm 
#else
    int 
#endif
    )
{
    adios_save_state *state = (adios_save_state *)funcData;

    // Map Conduit types to ADIOS types.
    ADIOS_DATATYPES dtype = conduit_dtype_to_adios_dtype(node);
    if(dtype == adios_unknown)
    {
        CONDUIT_ERROR("Unsupported Conduit to ADIOS type conversion.");
        return;
    }

    // Prepend a path if necessary.
    std::string adios_var(conduit::utils::join_path(state->adios_path, node.path()));

    // Dimensions
    char offset[20], global[20], local[20];
    memset(offset, 0, sizeof(char) * 20);
    memset(global, 0, sizeof(char) * 20);
    memset(local,  0, sizeof(char) * 20);
    if(node.dtype().number_of_elements() > 1)
    {
        sprintf(local, "%ld", node.dtype().number_of_elements());
    }
    else
    {
        sprintf(offset, "%d", internals::rank);
        sprintf(global, "%d", internals::size);
        sprintf(local,  "%d", 1);   
    }
    DEBUG_PRINT_RANK("adios_define_var(gid, \"" << adios_var
                     << "\", \"\", " << adios_type_to_string(dtype)
                     << ", \"" << local << "\""
                     << ", \"" << global << "\""
                     << ", \"" << offset << "\")")

    int64_t vid = adios_define_var(state->gid, adios_var.c_str(), 
                      "", dtype,
                      local, global, offset);
    if(vid < 0)
    {
        CONDUIT_ERROR("ADIOS Error:" << adios_get_last_errmsg()); 
        return;
    }

    // If we wanted a data transform in the options, add that now.
    const int transform_threshold = 1;
    if(!options()->transform.empty() &&
       node.dtype().number_of_elements() > transform_threshold)
    {
        std::vector<std::string> transform, tmp;
        if(options()->transform == "zfp")
        {
            // Only allow zfp on floating point variables.
            if(dtype == adios_real || dtype == adios_double)
            {
                // The zfp transform complained about passing multiple options at once.
                conduit::utils::split_string(options()->transform_options, ',', tmp);
                for(size_t i = 0; i < tmp.size(); i++)
                    transform.push_back(options()->transform + ":" + tmp[i]);
            }
        }
        else
        {
            transform.push_back(options()->transform + ":" + 
                                options()->transform_options);
        }

        for(size_t i = 0; i < transform.size(); ++i)
        {
            DEBUG_PRINT_RANK("adios_set_transform(varid, "
                             << "\"" << transform[i] << "\")")
            if(adios_set_transform(vid, transform[i].c_str()) != 0)
            {
                CONDUIT_ERROR("ADIOS Error:" << adios_get_last_errmsg()); 
            }
        }
    }

    // Store the name of the actual Conduit type as an attribute.
    DEBUG_PRINT_RANK("adios_define_attribute(gid, \"conduit_type\", \""
                     << adios_var << "\", adios_string, "
                     << "\"" << node.dtype().name() << "\", \""
                     << adios_var << "\")")
    if(adios_define_attribute(state->gid, "conduit_type", adios_var.c_str(),
        adios_string, node.dtype().name().c_str(), adios_var.c_str()) < 0)
    {
        CONDUIT_ERROR("ADIOS Error:" << adios_get_last_errmsg()); 
        return;
    }

    // Add the variable's var size to the total for the group.
    state->gSize += adios_expected_var_size(vid);
}

//-----------------------------------------------------------------------------
static void write_variables(const Node &node, void *funcData,
#ifdef USE_MPI
    MPI_Comm
#else
    int
#endif
    )
{
    adios_save_state *state = (adios_save_state *)funcData;

    // Map Conduit types to ADIOS types.
    ADIOS_DATATYPES dtype = conduit_dtype_to_adios_dtype(node);
    if(dtype == adios_unknown)
    {
        CONDUIT_ERROR("Unsupported Conduit to ADIOS type conversion.");
        return;
    }
    DEBUG_PRINT_RANK("adios_write(file, \"" << node.path()
                     << "\", node.data_ptr())")

    // Prepend a path if necessary.
    std::string adios_var(conduit::utils::join_path(state->adios_path, node.path()));

    // if the node is compact, we can write directly from its data ptr
    int s;
    if(node.dtype().is_compact()) 
    {
        s = adios_write(state->fid, adios_var.c_str(), node.data_ptr());
    }
    else
    {
        // otherwise, we need to compact our data first
        Node n;
        node.compact_to(n);
        s = adios_write(state->fid, adios_var.c_str(), n.data_ptr());
    }

    if(s != 0)
    {
        CONDUIT_ERROR("ADIOS Error: " << adios_get_last_errmsg());
    }
}

//-----------------------------------------------------------------------------
static bool declare_group(int64_t *gid, 
    const std::string &groupName, const std::string &timeIndex)
{
    DEBUG_PRINT_RANK("adios_declare_group(&gid, \""
        << groupName << "\", \"" << timeIndex << "\", "
        << internals::statistics_flag_to_string(options()->statistics_flag)
        << ")")

    int retval = adios_declare_group(gid, groupName.c_str(), timeIndex.c_str(), 
        options()->statistics_flag);
    if(retval != 0)
    {
        CONDUIT_ERROR("ADIOS error: " << adios_get_last_errmsg());
        return false;
    }

    const char *base_path = ""; // blank for current directory.
    DEBUG_PRINT_RANK("adios_select_method(gid, \"" 
        << options()->transport << "\", "
        << "\"" << options()->transport_options << "\", "
        << "\"" << base_path << "\")")
    retval = adios_select_method(*gid,
                 options()->transport.c_str(),
                 options()->transport_options.c_str(),
                 base_path);
    if(retval != 0)
    {
        CONDUIT_ERROR("ADIOS error: " << adios_get_last_errmsg());
        return false;
    }

    return true;
}

//-----------------------------------------------------------------------------
static int save(const Node &node, const std::string &path, 
    adios_save_state *state, const char *flag,
#ifdef USE_MPI
    MPI_Comm comm
#else
    int comm
#endif
    )
{
    //
    // Open the file
    //
    DEBUG_PRINT_RANK("adios_open(&fid, \"" << state->groupName << "\", \"" << path << "\", "
                     << "\"" << flag << "\", comm)")
    if(adios_open(&state->fid, state->groupName, path.c_str(), flag, comm) != 0)
    {
        CONDUIT_ERROR("ADIOS error: " << adios_get_last_errmsg());
        return -1;
    }

    // This is an optional call that lets ADIOS size its output buffers.
    uint64_t total_size = 0;
    DEBUG_PRINT_RANK("adios_group_size(fid, " << state->gSize << ", &total)")
    if(adios_group_size(state->fid, state->gSize, &total_size) != 0)
    {
        CONDUIT_ERROR("ADIOS error: " << adios_get_last_errmsg());
        return -2;
    }

    //
    // Write Variables
    //
    iterate_conduit_node(node, write_variables, state, comm);

    //
    // Close the file.
    //
    DEBUG_PRINT_RANK("adios_close(fid)")
    if(adios_close(state->fid) != 0)
    {
//        CONDUIT_ERROR("ADIOS error: " << adios_get_last_errmsg());
        return -3;
    }

    return 0;
}

//-----------------------------------------------------------------------------
static void
save(const Node &node, const std::string &path, const char *flag
    CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm comm)
    )
{
    adios_save_state state;
    std::string file_path(path);

    // check for ":" split  
    conduit::utils::split_file_path(path,
                                    std::string(":"),
                                    file_path,
                                    state.adios_path);

#ifdef USE_MPI
    MPI_Comm_rank(comm, &internals::rank);
    MPI_Comm_size(comm, &internals::size);

    // Bcast the filename so we know it will be the same on all ranks.
    int len = path.size()+1;
    MPI_Bcast(&len, 1, MPI_INT, 0, comm);
    char *sbuf = new char[len];
    if(internals::rank == 0)
        strcpy(sbuf, path.c_str());
    MPI_Bcast(sbuf, len, MPI_CHAR, 0, comm);
    if(rank > 0)
        file_path = std::string(sbuf);
    delete [] sbuf;

    // Initialize ADIOS.
    initialize(comm);
#else
    // Initialize ADIOS.
    initialize();
#endif

    // Set ADIOS's max buffer sized based on the options.
    DEBUG_PRINT_RANK("adios_set_max_buffer_size("
                     << options()->buffer_size << ")")
    adios_set_max_buffer_size(static_cast<uint64_t>(options()->buffer_size));

    //
    // Group
    //
    create_group_name(state.groupName);
#if defined(USE_MPI) && defined(MAKE_SEPARATE_GROUPS)
    MPI_Bcast(state.groupName, 32, MPI_CHAR, 0, comm);
#endif
    std::string timeIndex;
    if(declare_group(&state.gid, state.groupName, timeIndex))
    {
        //
        // Define variables and save the data.
        //
#ifdef USE_MPI
        iterate_conduit_node(node, define_variables, &state, comm);
        save(node, file_path, &state, flag, comm);
#else
        iterate_conduit_node(node, define_variables, &state, 0);
        save(node, file_path, &state, flag, 0);
#endif
    }
    else
    {
        CONDUIT_ERROR("ADIOS Error: failed to create group.");
    }

    // Delete the variable definitions from the group so we can define them
    // again the next time around. Free the group too.
    DEBUG_PRINT_RANK("adios_delete_vardefs(gid)")
    adios_delete_vardefs(state.gid);
    DEBUG_PRINT_RANK("adios_free_group(gid)")
    adios_free_group(state.gid);
}

//-----------------------------------------------------------------------------
bool name_matches_subpaths(const std::string &name, 
    const std::vector<std::string> &subpaths)
{
    bool match = false;
    if(subpaths.empty())
        match = true;
    else
    {
        for(size_t i = 0; i < subpaths.size(); ++i)
        {
            size_t lsubpath = subpaths[i].size();
            if(strncmp(name.c_str(), subpaths[i].c_str(), lsubpath) == 0)
            {
                // We have a match so far.
                size_t lname = name.size();
                if(lname == lsubpath)
                {
                    // name:    a/b/c
                    // subpath: a/b/c
                    match = true;
                    break;
                }
                else if(lname < lsubpath)
                {
                    // name:    a/b/c
                    // subpath: a/b/cat
                    match = false;
                }
                else if(lname > lsubpath)
                {
                    // name:   a/b/cat, a/b/c/d
                    // subpath a/b/c
                    if(name[lsubpath] == '/')
                    {
                        match = true;
                        break;
                    }
                }
            }
        }
    }
    return match;
}

//-----------------------------------------------------------------------------
void read_conduit_type_attribute(ADIOS_FILE *afile,
    ADIOS_VARINFO *v, std::string &conduit_type)
{
    for(int q = 0; q < v->nattrs; ++q)
    {
        if(strstr(afile->attr_namelist[v->attr_ids[q]], "conduit_type") != NULL)
        {
            ADIOS_DATATYPES atype;
            int asize = 0;
            void *aptr = NULL;
            adios_get_attr_byid(afile, v->attr_ids[q], &atype, &asize, &aptr);
            if(atype == adios_string)
                conduit_type = std::string((const char *)aptr);
            free(aptr);
            break;
        }
    }
}

//-----------------------------------------------------------------------------
struct adios_load_state
{
   adios_load_state() : filename(), time_step(CURRENT_TIME_STEP), domain(0),
       subpaths()
   {
   }

   std::string              filename;
   int                      time_step;
   int                      domain; 
   std::vector<std::string> subpaths;
};

//-----------------------------------------------------------------------------
void
open_file_and_process(adios_load_state *state,
    void (*cb)(adios_load_state *, ADIOS_FILE *, void *),
    void *cbdata
    CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm comm)
    )
{
#ifdef USE_MPI
    MPI_Comm_rank(comm, &internals::rank);
    MPI_Comm_size(comm, &internals::size);
#endif

    // Initialize the read method.
    DEBUG_PRINT_RANK("adios_read_init_method(" 
       << internals::read_method_to_string(internals::options()->read_method)
       << ", comm, \""
       << internals::options()->read_parameters
       << "\")");
    adios_read_init_method(options()->read_method, 
#ifdef USE_MPI
                           comm,
#else
                           0,
#endif
                           options()->read_parameters.c_str()
                           );
    // Mark that we've initialized the read method.
    mark_read_method_initialized(options()->read_method);

    // Open the file for read.
    ADIOS_FILE *afile = NULL;

    if(streamIsFileBased(options()->read_method))
    {
        // NOTE: This read function seems to have much better behavior in
        //       providing access to time steps.
        afile = adios_read_open_file(state->filename.c_str(), 
                    options()->read_method,
#ifdef USE_MPI
                    comm
#else
                    0
#endif
                    );
    }
    else
    {
        // NOTE: use adios_read_open to open the stream.
        DEBUG_PRINT_RANK("adios_read_open(\"" << state->filename << "\","
            << internals::read_method_to_string(options()->read_method) << ", "
            << "comm, "
            << internals::lock_mode_to_string(options()->read_lock_mode) << ", "
            << options()->read_timeout << ")")

        afile = adios_read_open(state->filename.c_str(), 
                    options()->read_method,
#ifdef USE_MPI
                    comm,
#else
                    0,
#endif
                    options()->read_lock_mode,
                    options()->read_timeout);
    }

    if(afile != NULL)
    {
        // Now that the file is open, do something with it.
        (*cb)(state, afile, cbdata);

        // Close the file.
        DEBUG_PRINT_RANK("adios_read_close(afile)")
        adios_read_close(afile);
    }
    else
    {
        CONDUIT_ERROR("ADIOS Error: " << adios_get_last_errmsg());
    }
}

//-----------------------------------------------------------------------------
// Callback for load()
static void
load_node(adios_load_state *state, ADIOS_FILE *afile, void *cbdata)
{
    // Convert cbdata back to Node * and get a reference.
    Node *node_ptr = (Node *)cbdata;
    Node &node = *node_ptr;

    int scheduled_reads = 0;
    std::vector<ADIOS_SELECTION *> sels;

    for (int i = 0; i < afile->nvars; ++i)
    {
        std::string vname(afile->var_namelist[i]);

        // Skip ADIOS statistics variables.
        if(strncmp(afile->var_namelist[i], "/__adios__", 10) == 0)
            continue;

        // Test that the variable is something we want to read.
        if(!internals::name_matches_subpaths(vname, state->subpaths))
            continue;

        DEBUG_PRINT_RANK("adios_inq_var(afile, \""
            << afile->var_namelist[i] << "\")")
        ADIOS_VARINFO *v = adios_inq_var(afile, afile->var_namelist[i]);
        if(v)
        {
            // We have array data. Let's allocate a buffer for 
            // it and schedule a read.
            DEBUG_PRINT_RANK("adios_inq_var_blockinfo(afile, v)")
            adios_inq_var_blockinfo(afile,v);

            // Check time step validity.
            int ts = state->time_step;
            if(state->time_step == CURRENT_TIME_STEP)
                ts = afile->current_step;
//            else if(state->time_step < v->nsteps)
//                ts = state->time_step;

#if 0
            // Print variable information.
            if(internals::rank == 0)
            {
                cout << "Reading process_id=" << state->domain << " time_index=" << (ts+1) << endl;
                internals::print_varinfo(cout, afile, v);
            }
#endif

            // ADIOS time steps start at 1.
            uint32_t ts1 = static_cast<uint32_t>(ts) + 1;
//cout << "state->time_step = " << state->time_step << ", v->nsteps=" << v->nsteps
//     << ", ts=" << ts << ", ts1 = " << ts1 << endl;

            // The blocks for the current time step start at biOffset,
            // which is the sum of the preceding block counts.
            int biOffset = 0;
            for(int ti = 0; ti < ts; ++ti)
                biOffset += v->nblocks[ti];

            // There will be v->nblocks[ts] blocks for the current
            // time step starting at biOffset. We can make sure that
            // it's for the right time step. The blocks we ask for
            // in the read call take the time step and the local block
            // number in that time step. So, read_dom will be
            // 0..nblocks[ts]-1.
            bool block_found = false;
            uint64_t nelem = 1;
            int read_dom = state->domain;
            if(v->blockinfo[biOffset + state->domain].time_index == ts1)
            {
                nelem = v->blockinfo[biOffset + state->domain].count[0];
                if(nelem == 0)
                {
                    // The count is 0 so we probably have a scalar.
                    //  Use the v->dims instead.
                    nelem = 1;
                    for(int d = 0; d < v->ndim; ++d)
                        nelem *= std::max(uint64_t(1), v->dims[d]);
                }
                block_found = true;                    
            }

            if(block_found)
            {
                // Select the block we want. This number is local to the 
                // current time step.
                DEBUG_PRINT_RANK("adios_selection_writeblock(" << read_dom << ")")
                ADIOS_SELECTION *sel = adios_selection_writeblock(read_dom);
                sels.push_back(sel);

                // Allocate memory for the variable.
                void *vbuf = NULL;
                if(v->type == adios_byte)
                {
                    std::string conduit_type;
                    internals::read_conduit_type_attribute(afile, 
                        v, conduit_type);
                    if(!conduit_type.empty() &&
                       conduit_type == "char8_str")
                    {
                        // Treat the byte array as a string.
                        // NOTE: when the data were saved, a '\0' terminator
                        //       was saved with the string. Decrement nelem
                        //       by 1 since the string will add space for
                        //       the terminator. This way the string sizes
                        //       are right.
                        size_t n_minus_nullterm = nelem - 1;
                        node[vname].set(std::string(n_minus_nullterm, ' '));
                        vbuf = (void *)node[vname].data_ptr();
                    }
                    else
                    {
                        node[vname].set(DataType::int8(nelem));
                        int8 *buf = node[vname].value();
                        vbuf = (void *)buf;
                    }
                }
                else if(v->type == adios_short)
                {
                    node[vname].set(DataType::int16(nelem));
                    int16 *buf = node[vname].value();
                    vbuf = (void *)buf;
                }
                else if(v->type == adios_integer)
                {
                    node[vname].set(DataType::int32(nelem));
                    int32 *buf = node[vname].value();
                    vbuf = (void *)buf;
                }
                else if(v->type == adios_long)
                {
                    node[vname].set(DataType::int64(nelem));
                    int64 *buf = node[vname].value();
                    vbuf = (void *)buf;
                }
                else if(v->type == adios_unsigned_byte)
                {
                    node[vname].set(DataType::uint8(nelem));
                    uint8 *buf = node[vname].value();
                    vbuf = (void *)buf;
                }
                else if(v->type == adios_unsigned_short)
                {
                    node[vname].set(DataType::uint16(nelem));
                    uint16 *buf = node[vname].value();
                    vbuf = (void *)buf;
                }
                else if(v->type == adios_unsigned_integer)
                {
                    node[vname].set(DataType::uint32(nelem));
                    uint32 *buf = node[vname].value();
                    vbuf = (void *)buf;
                }
                else if(v->type == adios_unsigned_long)
                {
                    node[vname].set(DataType::uint64(nelem));
                    uint64 *buf = node[vname].value();
                    vbuf = (void *)buf;
                }
                else if(v->type == adios_real)
                {
                    node[vname].set(DataType::float32(nelem));
                    float32 *buf = node[vname].value();
                    vbuf = (void *)buf;
                }
                else if(v->type == adios_double)
                {
                    node[vname].set(DataType::float64(nelem));
                    float64 *buf = node[vname].value();
                    vbuf = (void *)buf;
                }
                else
                {
                    // Other cases should not happen.
                    CONDUIT_ERROR("Unsupported type " << v->type);
                }

                // Schedule the read. NOTE that the time step passed here will
                // cooperate with the block selection to get the right block
                // for the right time step.
                if(vbuf != NULL)
                {
                    DEBUG_PRINT_RANK("adios_schedule_read_byid(afile, sel, "
                        << v->varid << ", " << ts << ", 1, vbuf)")
                    adios_schedule_read_byid(afile, sel, v->varid, ts, 1, vbuf);
                    scheduled_reads++;
                }
            } // block_found
#if 0
            else
            {
                // We could not find the desired block.
                cout << "No block for " << vname << " process_id=" << state->domain
                     << " time_index=" << (ts+1) << endl;
                internals::print_varinfo(cout, afile, v);
            }
#endif
            adios_free_varinfo(v);
        }           
    }

    // Perform any outstanding reads, blocking until reads are done.
    if(scheduled_reads > 0)
    {
        int blocking = 1;
        DEBUG_PRINT_RANK("adios_perform_reads(afile, " << blocking << ")")
        adios_perform_reads(afile, blocking);
    }
#if 0
    // Free the selections. (TODO: see if this still crashes ADIOS)
    for(size_t s = 0; s < sels.size(); ++s)
        adios_selection_delete(sels[s]);
#endif
}

//-----------------------------------------------------------------------------
static void
load(adios_load_state *state, Node *node
    CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm comm)
    )
{
    open_file_and_process(state, load_node, node
#ifdef USE_MPI
        ,comm
#endif
        );
}

//-----------------------------------------------------------------------------
// Callback for query_number_of_domains()
static void
find_max_domains(adios_load_state *state, ADIOS_FILE *afile, void *cbdata)
{
    int *ndoms = (int *)cbdata;

    for (int i = 0; i < afile->nvars; ++i)
    {
        std::string vname(afile->var_namelist[i]);

        // Skip ADIOS statistics variables.
        if(strncmp(afile->var_namelist[i], "/__adios__", 10) == 0)
            continue;

        // Test that the variable is something we want to read.
        if(!internals::name_matches_subpaths(vname, state->subpaths))
            continue;
        DEBUG_PRINT_RANK("adios_inq_var(afile, \""
            << afile->var_namelist[i] << "\")")

        ADIOS_VARINFO *v = adios_inq_var(afile, afile->var_namelist[i]);
        if(v)
        {
            DEBUG_PRINT_RANK("adios_inq_var_blockinfo(afile, v)")
            adios_inq_var_blockinfo(afile,v);

             // This determines which nblocks we'll look at.
            int ts = state->time_step;
            if(state->time_step == CURRENT_TIME_STEP)
                ts = afile->current_step;
            else if(state->time_step >= v->nsteps)
                ts = v->nsteps-1;

            // Let's look at the nblocks for the current time step.
            *ndoms = std::max(*ndoms, v->nblocks[ts]);
     
            adios_free_varinfo(v);
        }
    }
}

//-----------------------------------------------------------------------------
static int
query_number_of_domains(adios_load_state *state
    CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm comm)
   )
{ 
    int ndoms = 0;
    open_file_and_process(state, find_max_domains, &ndoms
#ifdef USE_MPI
        ,comm
#endif
        );

    return ndoms;
}

//-----------------------------------------------------------------------------
// Callback for query_number_of_time_steps()
static void
find_num_timesteps(adios_load_state *, ADIOS_FILE *afile, void *cbdata)
{
    int *ntimesteps = (int *)cbdata;
    // last_step is the index of the time. To get the number of time steps, add 1.
    *ntimesteps = afile->last_step + 1;
}


//-----------------------------------------------------------------------------
static int
query_number_of_time_steps(adios_load_state *state
    CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm comm)
   )
{ 
    int ntimesteps = 0;
    open_file_and_process(state, find_num_timesteps, &ntimesteps
#ifdef USE_MPI
        ,comm
#endif
        );

    return ntimesteps;
}


} // namespace
//-----------------------------------------------------------------------------
// -- end conduit::relay::<mpi>::io::internals --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
void
adios_set_options(const Node &opts
    CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm comm)
    )
{
#ifdef USE_MPI
    internals::initialize(comm);
#else
    internals::initialize();
#endif
    internals::options()->set(opts);
}

//-----------------------------------------------------------------------------
void
adios_options(Node &opts
    CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm comm)
    )
{
#ifdef USE_MPI
    internals::initialize(comm);
#else
    internals::initialize();
#endif
    internals::options()->about(opts);
}

//-----------------------------------------------------------------------------
void
adios_initialize_library(
    CONDUIT_RELAY_COMMUNICATOR_ARG0(MPI_Comm comm)
    )
{
#ifdef USE_MPI
    internals::initialize(comm);
#else
    internals::initialize();
#endif
}

//-----------------------------------------------------------------------------
void
adios_finalize_library(
    CONDUIT_RELAY_COMMUNICATOR_ARG0(MPI_Comm comm)
    )
{
#ifdef USE_MPI
    internals::finalize(comm);
#else
    internals::finalize();
#endif
}

//-----------------------------------------------------------------------------
void
adios_save(const Node &node, const std::string &path
    CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm comm)
    )
{
#ifdef USE_MPI
    internals::save(node, path, "w", comm);
#else
    internals::save(node, path, "w");
#endif
}

//-----------------------------------------------------------------------------
void adios_save_merged(const Node &node, const std::string &path
   CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm comm)
   )
{
    // NOTE: we use "u" to update the file so the time step is not incremented.
#ifdef USE_MPI
    internals::save(node, path, "u", comm);
#else
    internals::save(node, path, "u");
#endif
}

//-----------------------------------------------------------------------------
void adios_add_time_step(const Node &node, const std::string &path
    CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm comm))
{
    // check for ":" split  
    std::string file_path, adios_path;
    conduit::utils::split_file_path(path,
                                    std::string(":"),
                                    file_path,
                                    adios_path);

    // NOTE: we use "a" to update the file to the next time step.
#ifdef USE_MPI
    internals::save(node, path, "a", comm);
#else
    internals::save(node, path, "a");
#endif
}

//-----------------------------------------------------------------------------
void
adios_load(const std::string &path,
   int time_step,
   int domain, 
   Node &node
   CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm comm)
   )
{
    internals::adios_load_state state;
    state.time_step = time_step; // Force specific timestep/domain.
    state.domain = domain;

    // Split the incoming path in case it includes other information.
    // This may override the timestep and domain.
    internals::splitpath(path, state.filename, state.time_step, state.domain, state.subpaths);

#ifdef USE_MPI
    internals::load(&state, &node, comm);
#else
    internals::load(&state, &node);
#endif
}

//-----------------------------------------------------------------------------
void
adios_load(const std::string &path, Node &node
   CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm comm)
   )
{
    internals::adios_load_state state;
#ifdef USE_MPI
    // Read the rank'th domain if there is one.
    MPI_Comm_rank(comm, &state.domain);
#endif

    // Split the incoming path in case it includes other information.
    // This may override the timestep and domain.
    internals::splitpath(path, state.filename, state.time_step, state.domain, state.subpaths);

#ifdef USE_MPI
    internals::load(&state, &node, comm);
#else
    internals::load(&state, &node);
#endif
}

//-----------------------------------------------------------------------------
int
adios_query_number_of_time_steps(const std::string &path
    CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm comm)
   )
{
    internals::adios_load_state state;
    std::string tmp;

    // check for ":" split  
    conduit::utils::split_file_path(path,
                                    std::string(":"),
                                    state.filename,
                                    tmp);

#ifdef USE_MPI
    return internals::query_number_of_time_steps(&state, comm);
#else
    return internals::query_number_of_time_steps(&state);
#endif
}

//-----------------------------------------------------------------------------
int
adios_query_number_of_domains(const std::string &path
    CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm comm)
   )
{
    internals::adios_load_state state;

    // Split the incoming path in case it includes other information.
    // This may override the timestep and domain.
    internals::splitpath(path, state.filename, state.time_step, state.domain, state.subpaths);

#ifdef USE_MPI
    return internals::query_number_of_domains(&state, comm);
#else
    return internals::query_number_of_domains(&state);
#endif
}

}
//-----------------------------------------------------------------------------
// -- end conduit::relay::<mpi>::io --
//-----------------------------------------------------------------------------

#ifdef USE_MPI
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
