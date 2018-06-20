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

// NOTE: We can save strings to ADIOS as adios_string if they are scalars.
//       However, when we write adios_string in parallel, it seems as though
//       we get multiple chunks of data stored in the file but accessing the
//       strings beyond the first chunk seems problematic. So, we have trouble
//       reading the string contributions from later ranks. A workaround is to
//       save the data as adios_byte since that produces an array we can access
//       as intended. We use the conduit_type attribute that we put on each
//       variable as a hint to indicate that while the data are adios_byte,
//       we should reconstitute them as a string.
#define CONDUIT_ADIOS_STRING_AS_BYTE_ARRAY

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
static int size = 0;

#define DEBUG_PRINT_RANK(OSSEXPR) if(internals::rank == 0) { cout << OSSEXPR << endl; }
//#define DEBUG_PRINT_RANK(OSSEXPR) 

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
    return s;
}

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
    return m;
}

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

std::string
statistics_flag_to_string(ADIOS_STATISTICS_FLAG f)
{
    std::string s("adios_stat_default");
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

ADIOS_STATISTICS_FLAG
string_to_statistics_flag(const std::string &s)
{
    ADIOS_STATISTICS_FLAG f = adios_stat_default;
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

void
split_string(std::vector<std::string> &sv, const std::string &s, char sep)
{
    if(!s.empty())
    {
        const char *start = s.c_str();
        const char *c     = s.c_str();
        while(*c != '\0')
        {
            if(*c == sep)
            {
                size_t len = c - start;
                if(len > 0)
                    sv.push_back(std::string(start, len));
                c++;
                start = c;
            }
            else
                c++;
        }
        if(*start != '\0')
        {
            size_t len = c - start;
            if(len > 0)
                sv.push_back(std::string(start, len));
        }
    }
}

} // namespace
//-----------------------------------------------------------------------------
// -- end conduit::relay::<mpi>::io::internals --
//-----------------------------------------------------------------------------

static bool adiosState_initialized = false;

//-----------------------------------------------------------------------------
static void conduit_adios_initialize(
#ifdef USE_MPI
    MPI_Comm comm
#else
    int comm
#endif
    )
{
    cout << "conduit_adios_initialize: init'd=" << adiosState_initialized << endl;
    if(!adiosState_initialized)
    {
#ifdef USE_MPI
        // See if MPI is initialized.
        int mpi_init = 0;
        MPI_Initialized(&mpi_init);
        cout << "mpi_init = " << mpi_init << endl;

        // Initialize ADIOS.
        int status = adios_init_noxml(comm);
        cout << "adios_init_noxml = " << status << endl;
#else
        cout << "initializing serial ADIOS." << endl;
        // Initialize ADIOS.
        int status = adios_init_noxml(comm);
        cout << "adios_init_noxml = " << status << endl;
#endif
        adiosState_initialized = true;
    }
}

//-----------------------------------------------------------------------------
static bool valid_transport(const std::string &name)
{
    bool valid = false;
    ADIOS_AVAILABLE_WRITE_METHODS *wm = adios_available_write_methods();
    if(wm)
    {
        for (int i=0; i<wm->nmethods; i++)
        {
            if(strcmp(name.c_str(), wm->name[i]) == 0)
            {
                valid = true;
                break;
            }
        }
        adios_available_write_methods_free(wm); 
    }
    return valid;
}

//-----------------------------------------------------------------------------
static std::vector<std::string> conduit_adios_transports()
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
static bool valid_transform(const std::string &name)
{
    bool valid = false;
#if 0 // Crashing
    ADIOS_AVAILABLE_TRANSFORM_METHODS * t = adios_available_transform_methods();
    if(t)
    {
        for (int i=0; i<t->ntransforms; i++)
        {
            if(strcmp(name.c_str(), t->name[i]) == 0)
            {
                valid = true;
                break;
            }
        }
        adios_available_transform_methods_free(t); 
    }
#endif
    return valid;
}

//-----------------------------------------------------------------------------
static std::vector<std::string> conduit_adios_transforms()
{
    std::vector<std::string> v;
#if 0 // Crashing
    ADIOS_AVAILABLE_TRANSFORM_METHODS * t = adios_available_transform_methods();
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
static std::string 
join_string_vector(const std::vector<std::string> &sv, const std::string &sep)
{
    std::string s;
    for(size_t i=0; i<sv.size(); i++)
    {
        if(i > 0)
            s = s + sep;
        s = s + std::string(sv[i]);
    }
    return s;
}

#define UNINITIALIZED_COMM 0

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
        buffer_size(1024*1024), transport("POSIX"), 
        statistics_flag(adios_stat_default), transform(),
        transport_options(),
        transform_options(),

        // Read options
        read_method(ADIOS_READ_METHOD_BP),
        read_parameters(),
        read_lock_mode(ADIOS_LOCKMODE_CURRENT),
        read_verbose(0),
        read_timeout(0.f)
    {
cout << "ADIOSOptions ctor start" << endl;
#ifdef USE_MPI
        int mpi_init = 0;
        MPI_Initialized(&mpi_init);
        cout << "mpi_init = " << mpi_init << endl;


        transport = "MPI";
cout << "ADIOSOptions ctor end" << endl;
#endif
    }
    
    //------------------------------------------------------------------------
    void set(const Node &opts)
    {
        // We need to initialize if we have not done so we can call
        // valid_transport and valid_transform.
#ifdef USE_MPI
        conduit_adios_initialize(MPI_COMM_WORLD);
#else
        conduit_adios_initialize(0);
#endif

        // Write options
        if(opts.has_child("write"))
        {
cout << "options set write" << endl;
            const Node &n = opts["write"];

            if(n.has_child("buffer_size"))
                buffer_size = opts["buffer_size"].as_long();

            if(n.has_child("transport"))
            {
                std::string s(n["transport"].as_string());
                if(valid_transport(s))
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
transform = s;

cout << "transform = " << transform << endl;

//                if(valid_transform(s))
//                    transform = s;
            }

            if(n.has_child("transport_options"))
                transport_options = n["transport_options"].as_string();

            if(n.has_child("transform_options"))
{
                transform_options = n["transform_options"].as_string();
cout << "transform_options = " << transform_options << endl;
}
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
                read_timeout = n["timeout"].as_float();

            if(n.has_child("verbose"))
                read_verbose = n["verbose"].to_value();
        }

#if 0
        Node tmp;
        about(tmp);
        cout << "ADIOS options: " << tmp.to_json() << endl;
#endif
    }

    //------------------------------------------------------------------------
    void about(Node &opts)
    {
        // We need to initialize if we have not done so we can call some
        // ADIOS introspection functions.
#ifdef USE_MPI
        conduit_adios_initialize(MPI_COMM_WORLD);
#else
        conduit_adios_initialize(0);
#endif

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
//
// NOTE: we might want to have write and read properties separate.
//
//      read/selection = all
//                       0-4
//                       0,1,2-5,7

        // Add in the available transports and transforms.
        std::string sep(", ");
        std::vector<std::string> transports(conduit_adios_transports()),
                                 transforms(conduit_adios_transforms());
        opts["information/available_transports"] = join_string_vector(transports, sep);
        opts["information/available_transforms"] = join_string_vector(transforms, sep);

/*
   adios/buffer_size = 2^20
   adios/group_name = "conduit"  <-- do this???
   adios/statistics_flag = "adios_stat_no", adios_stat_minmax, adios_stat_default, or adios_stat_full
   adios/transport = "MPI"


   adios/transports/POSIX/local_fs = 0
   adios/transports/MPI/verbose = 3
   adios/transports/MPI_LUSTRE/stripe_count=16
   adios/transports/MPI_LUSTRE/stripe_size=4194304
   adios/transports/MPI_LUSTRE/block_size=4194304

   adios/transports/MPI_AGGREGATE/num_aggregators = 24  // compute from lustre or ADIOS?
   adios/transports/mpi_aggregate/num_ost = 672
   adios/transports/mpi_aggregate/have_metadata_file = 0
   adios/transports/mpi_aggregate/striping = 0
   adios/transports/mpi_aggregate/stripe_count = 1
   adios/transports/mpi_aggregate/stripe_size = ?  // needs a good default
   adios/transports/MPI_AGGREGATE/random_offset = 1
   adios/transports/MPI_AGGREGATE/local_fs = 0

   adios/transports/VAR_MERGE/chunk_size = 22097152
   adios/transports/VAR_MERGE/io_method=MPI_AGGREGATE
   adios/transports/VAR_MERGE/io_parameters=24
   adios/transports/VAR_MERGE/num_aggregators=24
   adios/transports/VAR_MERGE/num_ost=672
   adios/transports/PHDF5/ ... are there transforms to enable?

   adios/transform = "zlib"

   adios/transforms/zlib/level = 5

   adios/transforms/lz4/threshold = 4096
   adios/transforms/lz4/level = 9

   adios/transforms/blosc/threshold = 4096
   adios/transforms/blosc/shuffle = "bit"  (no, bit, byte)
   adios/transforms/blosc/level = 1        [1,9]
   adios/transforms/blosc/threads = 4      [1,#]
   adios/transforms/blosc/compressor = "zstd"  (zlib, lz4, lz4hc, snappy, zstd, blosclz, memcpy)

   adios/transforms/zfp/rate = 0.25
   adios/transforms/zfp/precision = 16
   adios/transforms/zfp/accuracy = 0.0001

   adios/transforms/sz/absolute = 0.0001
   adios/transforms/sz/relative = 0.0001
   adios/transforms/sz/init = sz.config

   adios/read_only/available_transports = "MPI, MPI_LUSTRE, ..."
   adios/read_only/available_transforms = "zlib, lz4, blosc, ..."
   adios/read_only/version = "1.11.0"

*/

// TODO: transports can take parameters. See MPI_AMR. Look into this more.
// method="MPI_LUSTRE"> stripe_count=16,stripe_size=4194304,block_size=4194304

// TODO: expose a "time_index" so the user can pass a time step.

//       This would let the ADIOS library compute statistics that might be useful.
//       Q: How to send them to Conduit logs???
// TODO: "statistics_flag" = "adios_stat_no";
//                           adios_stat_minmax, adios_stat_default, or adios_stat_full
    }
};

// TODO: When MACSio initializes its plugins, it uses a static initialization and I've
//       observed that the transport std::string is not necessarily always initialized...
//       Maybe we need to allocate this on demand in an accessor function.

// default adios i/o settings
static ADIOSOptions *adiosState_options = NULL;

// @brief Clean up the options at exit.
static void CleanupOptions(void)
{
    if(adiosState_options != NULL)
    {
        delete adiosState_options;
        adiosState_options = NULL;
    }
}

// @brief Access the ADIOS save options, creating them first if needed. 
//        We create them on the heap to make sure that the object does
//        not fail to initialize statically.
static ADIOSOptions *GetOptions()
{
    if(adiosState_options == NULL)
    {
        adiosState_options = new ADIOSOptions;
        atexit(CleanupOptions);
    }
    return adiosState_options;
}

//-----------------------------------------------------------------------------
static void iterate_conduit_node(const conduit::Node &node,
    void (*func)(const conduit::Node &,
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
    adios_save_state() : fid(0), gid(0), gSize(0)
    {
    }

    int64_t  fid;
    int64_t  gid;
    uint64_t gSize;
};

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
#ifdef CONDUIT_ADIOS_STRING_AS_BYTE_ARRAY
            dtype = adios_byte;
#else
            dtype = adios_string;
#endif
        }
    }

    return dtype;
}

static void define_variables(const Node &node, void *funcData,
#ifdef USE_MPI
    MPI_Comm comm
#else
    int comm
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

    // NOTE: If I knew I had a Conduit tree that described a Blueprint uniform mesh
    //       or rectilinear mesh then I might be able to put more sensible things
    //       in for this since we'd know that the data are simple array chunks.
    //       Can't make that assumption for a general Conduit tree.
#if 1
    // Dimensions
    char offset[10], global[10], local[10];
    if(node.dtype().number_of_elements() > 1)
    {
        sprintf(local, "%ld", node.dtype().number_of_elements());
        global[0] = '\0';
        offset[0] = '\0';
    }
    else
    {
        sprintf(offset, "%d", internals::rank);
        sprintf(global, "%d", internals::size);
        sprintf(local,  "%d", 1);   
    }
    DEBUG_PRINT_RANK("adios_define_var(gid, \"" << node.path()
                     << "\", \"\", " << adios_type_to_string(dtype)
                     << ", \"" << local << "\""
                     << ", \"" << global << "\""
                     << ", \"" << offset << "\")")

    int64_t vid = adios_define_var(state->gid, node.path().c_str(), 
                      "", dtype,
                      local, global, offset);
#else
    // Dimensions
    char dimensions[20];
    const char *global_dimensions = "";
    const char *local_offsets = "";
    if(dtype != adios_string && node.dtype().number_of_elements() > 1)
        sprintf(dimensions, "%ld", node.dtype().number_of_elements());
    else
        dimensions[0] = '\0';

    // Define the variable.
    DEBUG_PRINT_RANK("adios_define_var(gid, \"" << node.path()
                     << "\", \"\", " << adios_type_to_string(dtype)
                     << ", \"" << dimensions << "\""
                     << ", \"" << global_dimensions << "\""
                     << ", \"" << local_offsets << "\")")
    int64_t vid = adios_define_var(state->gid, node.path().c_str(), 
                      "", dtype,
                      dimensions, global_dimensions, local_offsets);
#endif
    if(vid < 0)
    {
        CONDUIT_ERROR("ADIOS Error:" << adios_get_last_errmsg()); 
        return;
    }

    // If we wanted a data transform in the options, add that now.
    const int transform_threshold = 1;
    if(!GetOptions()->transform.empty() &&
       node.dtype().number_of_elements() > transform_threshold)
    {
        std::vector<std::string> transform, tmp;
        if(GetOptions()->transform == "zfp")
        {
            // The zfp transform complained about passing multiple options at once.
            internals::split_string(tmp, GetOptions()->transform_options, ',');
            for(size_t i = 0; i < tmp.size(); i++)
                transform.push_back(GetOptions()->transform + ":" + tmp[i]);
        }
        else
        {
            transform.push_back(GetOptions()->transform + ":" + 
                                GetOptions()->transform_options);
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

    // Store the name of the actual Conduit type as an attribute in case we 
    // need it to read.
    DEBUG_PRINT_RANK("adios_define_attribute(gid, \"conduit_type\", \""
                     << node.path() << "\", adios_string, "
                     << "\"" << node.dtype().name() << "\", \""
                     << node.path() << "\")")
    if(adios_define_attribute(state->gid, "conduit_type", node.path().c_str(),
        adios_string, node.dtype().name().c_str(), node.path().c_str()) < 0)
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

    // if the node is compact, we can write directly from its data ptr
    int s;
    if(node.dtype().is_compact()) 
    {
        s = adios_write(state->fid, node.path().c_str(), node.data_ptr());
    }
    else
    {
        // otherwise, we need to compact our data first
        Node n;
        node.compact_to(n);
        s = adios_write(state->fid, node.path().c_str(), n.data_ptr());
    }

    if(s != 0)
    {
        CONDUIT_ERROR("ADIOS Error: " << adios_get_last_errmsg());
    }
}

//-----------------------------------------------------------------------------
static bool conduit_adios_declare_group(int64_t *gid, 
    const std::string &groupName, const std::string &timeIndex)
{
    DEBUG_PRINT_RANK("adios_declare_group(&gid, \""
        << groupName << "\", \"" << timeIndex << "\", "
        << internals::statistics_flag_to_string(GetOptions()->statistics_flag)
        << ")")
    int retval = adios_declare_group(gid, groupName.c_str(), timeIndex.c_str(), 
        GetOptions()->statistics_flag);
    if(retval != 0)
    {
        CONDUIT_ERROR("ADIOS error: " << adios_get_last_errmsg());
        return false;
    }

    const char *base_path = ""; // blank for current directory.
    DEBUG_PRINT_RANK("adios_select_method(gid, \"" 
        << GetOptions()->transport << "\", "
        << "\"" << GetOptions()->transport_options << "\", "
        << "\"" << base_path << "\")")
    retval = adios_select_method(*gid,
                 GetOptions()->transport.c_str(),
                 GetOptions()->transport_options.c_str(),
                 base_path);
    if(retval != 0)
    {
        CONDUIT_ERROR("ADIOS error: " << adios_get_last_errmsg());
        return false;
    }

    return true;
}

//-----------------------------------------------------------------------------
static int conduit_adios_save(const Node &node, const std::string &path, 
    adios_save_state *state, const char *flag,
#ifdef USE_MPI
    MPI_Comm comm
#else
    int comm
#endif
    )
{
    // NOTE: We have already done the group and variable declarations.
    // NOTE: Assume the path will be the same on all ranks in the comm.

    //
    // Open the file
    //
    DEBUG_PRINT_RANK("adios_open(&fid, \"conduit\", \"" << path << "\", "
                     << "\"" << flag << "\", comm)")
    if(adios_open(&state->fid, "conduit", path.c_str(), flag, comm) != 0)
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

    // Q: Do we need to store some schema in the file to make ADIOS->Conduit 
    //    reconstruction easier?

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
        CONDUIT_ERROR("ADIOS error: " << adios_get_last_errmsg());
        return -3;
    }

    return 0;
}

//-----------------------------------------------------------------------------
void
adios_set_options(const Node &opts)
{
    GetOptions()->set(opts);
}

//-----------------------------------------------------------------------------
void
adios_options(Node &opts)
{
    GetOptions()->about(opts);
}

//-----------------------------------------------------------------------------
void
adios_save(const Node &node, const std::string &path
   CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm comm)
   )
{
    std::string filename(path);

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
    if(internals::rank > 0)
        filename = std::string(sbuf);
    delete [] sbuf;

    // Initialize ADIOS.
    conduit_adios_initialize(comm);
#else
    // Initialize ADIOS.
    conduit_adios_initialize(0);
#endif

    // Set ADIOS's max buffer sized based on the options.
    DEBUG_PRINT_RANK("adios_set_max_buffer_size("
                     << GetOptions()->buffer_size << ")")
    adios_set_max_buffer_size(static_cast<uint64_t>(GetOptions()->buffer_size));
    adios_save_state state;
    state.fid = 0;
    state.gid = 0;
    state.gSize = 0;

    //
    // Group
    //
    std::string groupName("conduit"), timeIndex;
    if(conduit_adios_declare_group(&state.gid, groupName, timeIndex))
    {
        //
        // Define variables and save the data.
        //
#ifdef USE_MPI
        iterate_conduit_node(node, define_variables, &state, comm);
        conduit_adios_save(node, filename, &state, "w", comm);
#else
        iterate_conduit_node(node, define_variables, &state, 0);
        conduit_adios_save(node, filename, &state, "w", 0);
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
void adios_append(const Node &node, const std::string &path
   CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm comm)
   )
{
    cout << "conduit::relay::io::adios_append(node, path=" << path << ")" << endl;
}

//-----------------------------------------------------------------------------
bool name_matches_subtree(const std::string &name, const std::string &pattern)
{
    // for now.
    return true;
}

//-----------------------------------------------------------------------------
void conduit_adios_read_conduit_type_attribute(ADIOS_FILE *afile,
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
void adios_load(const std::string &path, Node &node, int time_step, int domain
   CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm comm)
   )
{
// NOTE: If we're loading a dataset and we give it a path, we want to load 
//       just this processor's piece...
//       For reading, that's not generally true as we might want to use a serial
//       read library to read separate domains that were all saved to the same file
//       and we'd want control over which parts we read out.

// FOR NOW
    std::string pattern;

#ifdef USE_MPI
    MPI_Comm_rank(comm, &internals::rank);
#endif

    // once per program run...
    DEBUG_PRINT_RANK("adios_read_init_method()");
    adios_read_init_method(GetOptions()->read_method, 
#ifdef USE_MPI
                           comm,
#else
                           0,
#endif
                           GetOptions()->read_parameters.c_str()
                           );

    // Open the file for read.
    DEBUG_PRINT_RANK("adios_read_open(\"" << path << "\","
        << internals::read_method_to_string(GetOptions()->read_method) << ", "
        << "comm, "
        << internals::lock_mode_to_string(GetOptions()->read_lock_mode) << ", "
        << GetOptions()->read_timeout << ")")
    ADIOS_FILE *afile = adios_read_open(path.c_str(), 
                            GetOptions()->read_method,
#ifdef USE_MPI
                            comm,
#else
                            0,
#endif
                            GetOptions()->read_lock_mode,
                            GetOptions()->read_timeout);
    if(afile != NULL)
    {
        int scheduled_reads = 0;
        std::vector<ADIOS_SELECTION *> sels;

        // timestep and domain args...
        int ts = (time_step == -1) ? afile->current_step : time_step;

        for (int i = 0; i < afile->nvars; ++i)
        {
            std::string vname(afile->var_namelist[i]);

            // Skip ADIOS statistics variables.
            if(strncmp(afile->var_namelist[i], "/__adios__", 10) == 0)
                continue;

            // Test that the variable is something we want to read.
            if(!name_matches_subtree(vname, pattern))
                continue;

            DEBUG_PRINT_RANK("adios_inq_var(afile, \""
                << afile->var_namelist[i] << "\")")
            ADIOS_VARINFO *v = adios_inq_var(afile, afile->var_namelist[i]);
            if(v)
            {
#if 0
                if(v->ndim == 0)
                {
// NOTE: if multiple ranks wrote out their piece, how can the data be in the v object?
//       Do we need a selection applied here already?

                    // scalar. The data value is already in the v object.
                    if(v->type == adios_byte)
                        node[vname] = *((int8*)v->value);
                    else if(v->type == adios_short)
                        node[vname] = *((int16*)v->value);
                    else if(v->type == adios_integer)
                        node[vname] = *((int32*)v->value);
                    else if(v->type == adios_long)
                        node[vname] = *((int64*)v->value);
                    else if(v->type == adios_unsigned_byte)
                        node[vname] = *((uint8*)v->value);
                    else if(v->type == adios_unsigned_short)
                        node[vname] = *((uint16*)v->value);
                    else if(v->type == adios_unsigned_integer)
                        node[vname] = *((uint32*)v->value);
                    else if(v->type == adios_unsigned_long)
                        node[vname] = *((uint64*)v->value);
                    else if(v->type == adios_real)
                        node[vname] = *((float32*)v->value);
                    else if(v->type == adios_double)
                        node[vname] = *((float64*)v->value);
                    else if(v->type == adios_string)
                    {
#if 0
                        std::string s;
                        s.assign((char *)v->value);
                        node[vname] = s;
#else
                        node[vname] = std::string((const char *)v->value);
#endif
                    }
                    // These cases should not happen.
                    else if(v->type == adios_complex)
                    {
                        CONDUIT_ERROR("Skipping adios_complex " << vname);
                    }
                    else if(v->type == adios_double_complex)
                    {
                        CONDUIT_ERROR("Skipping adios_double_complex " << vname);
                    }
                    else if(v->type == adios_string_array)
                    {
                        CONDUIT_ERROR("Skipping adios_string_array " << vname);
                    }
                    else if(v->type == adios_unknown)
                    {
                        CONDUIT_ERROR("Skipping adios_unknown " << vname);
                    }
#if 1
                    const Node &n = node[vname];
                    cout << rank << ": " << vname <<" = " << n.to_json() << endl;
#endif
                }
                else
                {
#endif
                    // We have array data. Let's allocate a buffer for 
                    // it and schedule a read.
                    DEBUG_PRINT_RANK("adios_inq_var_blockinfo(afile, v)")
                    adios_inq_var_blockinfo(afile,v);

                    // The number of blocks in the current step. Make sure that
                    // the domain is in range.
                    int nblocks = v->nblocks[ts];
                    int dom = domain;
                    if(domain < 0)
                        domain = 0;
                    if(domain > nblocks-1)
                        domain = nblocks-1;

#if 1
if(internals::rank == 0)
   internals::print_varinfo(cout, afile, v);
#endif


                    // Select the block we want.
                    DEBUG_PRINT_RANK("adios_selection_writeblock(" << dom << ")")
                    ADIOS_SELECTION *sel = adios_selection_writeblock(dom);
                    sels.push_back(sel);

                    // Use the dims to figure out the number of elements.
                    uint64_t nelem = 1;
                    if(v->sum_nblocks == 1)
                    {
                        for(int d = 0; d < v->ndim; ++d)
                            nelem *= std::max(uint64_t(1), v->dims[d]);
                    }
                    else
                    {
                        // We can't rely on the v->dims being good for anything 
                        // but the first MPI rank, at least the way we're writing
                        // the data. Compute the number of elements from the
                        // blockinfo so we can properly size the destination array.

                        // NOTE: can we assume these will be in process order?
                        int ts1 = ts + 1;
                        for(int bi = 0; bi < v->sum_nblocks; ++bi)
                        {
                            if(v->blockinfo[bi].time_index == ts1 &&
                               v->blockinfo[bi].process_id == domain)
                            {
                                nelem = v->blockinfo[bi].count[0];
                                break;
                            }
                        }
                    }

                    // Allocate memory for the variable.
                    void *vbuf = NULL;
                    if(v->type == adios_byte)
                    {
#ifdef CONDUIT_ADIOS_STRING_AS_BYTE_ARRAY
                        std::string conduit_type;
                        conduit_adios_read_conduit_type_attribute(afile, 
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

//cout << "buf = " << (void*)buf << endl;
//cout << "data_ptr = " << (void*)node[vname].data_ptr() << endl;

                        }
                        else
                        {
#endif
                            node[vname].set(DataType::int8(nelem));
                            int8 *buf = node[vname].value();
                            vbuf = (void *)buf;
#ifdef CONDUIT_ADIOS_STRING_AS_BYTE_ARRAY
                        }
#endif
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
#if 0
// We are not using adios_string for storing std::string
                    else if(v->type == adios_string)
                    {
cout << rank << ": adios_string: nelem=" << nelem << endl;
                        //int slen = strlen((char *)v->value); // BAD
                        //nelem = slen;
                        size_t n_minus_nullterm = nelem - 1;
                        node[vname].set(std::string(n_minus_nullterm, ' '));
                        vbuf = (void *)node[vname].data_ptr();
                    }
#endif
                    else
                    {
                        // Other cases should not happen.
                        CONDUIT_ERROR("Unsupported type " << v->type);
                    }

                    // Schedule the read.
                    if(vbuf != NULL)
                    {
                        DEBUG_PRINT_RANK("adios_schedule_read_byid(afile, sel, "
                            << v->varid << ", " << ts << ", 1, vbuf)")
                        adios_schedule_read_byid(afile, sel, v->varid, ts, 1, vbuf);
                        scheduled_reads++;
                    }
#if 0
                }
#endif

                adios_free_varinfo(v);
            }           
        }
//#ifdef USE_MPI
//MPI_Barrier(comm);
//#endif

        // Perform any outstanding reads, blocking until reads are done.
        if(scheduled_reads > 0)
        {
            int blocking = 1;
            DEBUG_PRINT_RANK("adios_perform_reads(afile, " << blocking << ")")
            adios_perform_reads(afile, blocking);
        }
#if 0
        // Free the selections.
        for(size_t s = 0; s < sels.size(); ++s)
            adios_selection_delete(sels[s]);
#endif

        // Close the file.
        DEBUG_PRINT_RANK("adios_read_close(afile)")
        adios_read_close(afile);
    }
    else
    {
        CONDUIT_ERROR("ADIOS Error: " << adios_get_last_errmsg());
    }

    // once per program run.
    DEBUG_PRINT_RANK("adios_read_finalize_method("
        << internals::read_method_to_string(GetOptions()->read_method)
        << ")")
    adios_read_finalize_method(GetOptions()->read_method);
}

//-----------------------------------------------------------------------------
void adios_load(const std::string &path, Node &node
   CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm comm)
   )
{
    int domain = 0;     // use first domain
    int time_step = -1; // use current time step.
#ifdef USE_MPI
    // Read the rank'th domain if there is one.
    MPI_Comm_rank(comm, &domain);
    adios_load(path, node, time_step, domain, comm);
#else
    adios_load(path, node, time_step, domain);
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
