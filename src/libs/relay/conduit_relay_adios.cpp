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

#include "conduit_relay_adios.hpp"

//-----------------------------------------------------------------------------
// standard lib includes
//-----------------------------------------------------------------------------
#include <iostream>

//-----------------------------------------------------------------------------
// external lib includes
//-----------------------------------------------------------------------------
#include <adios.h>
#include <adios_transform_methods.h>
#ifndef _NOMPI
    #include <mpi.h>
#endif

//-----------------------------------------------------------------------------
// -- conduit includes -- 
//-----------------------------------------------------------------------------
#include "conduit_error.hpp"
//#include "conduit_utils.hpp"

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


//-----------------------------------------------------------------------------
// -- begin conduit::relay::io --
//-----------------------------------------------------------------------------
namespace io
{

static bool adiosState_initialized = false;
static int  adiosState_mpicomm = -1;

//-----------------------------------------------------------------------------
static void conduit_adios_initialize(int mpicomm)
{
    std::cout << "conduit_adios_initialize: mpicomm=" << mpicomm << ", init'd=" << adiosState_initialized << std::endl;
    if(adiosState_initialized)
    {
        if(adiosState_mpicomm != mpicomm)
        {
             // TODO: do we reinitialize using another communicator?
             adiosState_initialized = false;
        }
    }

    if(!adiosState_initialized)
    {
        // Initialize ADIOS.
#ifndef _NOMPI
        std::cout << "conduit_adios_initialize: mpicomm=" << mpicomm << std::endl;
        adios_init_noxml(MPI_Comm_f2c(mpicomm));
#else
        adios_init_noxml(0);
#endif
        adiosState_initialized = true;
        adiosState_mpicomm = mpicomm;
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
    int                   mpicomm;
    bool                  collective;
    long                  buffer_size;
    std::string           transport;
    ADIOS_STATISTICS_FLAG statistics_flag;
    std::string           transform;
#ifdef SEPARATE_OPTIONS
    int POSIX_localfs;

    int MPI_verbose;

    int MPI_LUSTRE_stripe_count;
    int MPI_LUSTRE_stripe_size;
    int MPI_LUSTRE_block_count;

    int MPI_AGGREGATE_num_aggregators;
    int MPI_AGGREGATE_num_ost;
    int MPI_AGGREGATE_have_metadata_file;
    int MPI_AGGREGATE_striping;
    int MPI_AGGREGATE_stripe_count;
    int MPI_AGGREGATE_stripe_size;
    int MPI_AGGREGATE_random_offset;
    int MPI_AGGREGATE_localfs;

    int VAR_MERGE_chunk_size;
    std::string VAR_MERGE_io_method;
    int VAR_MERGE_io_parameters;
    int VAR_MERGE_num_aggregators;
    int VAR_MERGE_num_ost;
#else
    std::string transport_options;
    std::string transform_options;
#endif
public:
    ADIOSOptions() : mpicomm(0), collective(true), 
        buffer_size(1024*1024), transport("POSIX"), 
        statistics_flag(adios_stat_default), transform(),

#ifdef SEPARATE_OPTIONS
        // Transport-specific options. These values are from the ADIOS manual.
        // For some, it'd probably be better to detect some good values from
        // Lustre, etc.
        POSIX_localfs(0),

        MPI_verbose(0),

        MPI_LUSTRE_stripe_count(16),
        MPI_LUSTRE_stripe_size(4194304),
        MPI_LUSTRE_block_count(4194304),

        MPI_AGGREGATE_num_aggregators(24), // Again, machine-specific.
        MPI_AGGREGATE_num_ost(672),  
        MPI_AGGREGATE_have_metadata_file(0),
        MPI_AGGREGATE_striping(1),
        MPI_AGGREGATE_stripe_count(16),
        MPI_AGGREGATE_stripe_size(4194304),
        MPI_AGGREGATE_random_offset(1),
        MPI_AGGREGATE_localfs(0),

        VAR_MERGE_chunk_size(22097152),
        VAR_MERGE_io_method(MPI_AGGREGATE),
        VAR_MERGE_io_parameters(24),
        VAR_MERGE_num_aggregators(24),
        VAR_MERGE_num_ost(672)
#else
        transport_options(),
        transform_options()
#endif
    {
std::cout << "ADIOSOptions ctor start" << std::endl;
#ifndef _NOMPI
        mpicomm = MPI_Comm_c2f(MPI_COMM_WORLD);
        transport = "MPI";
std::cout << "ADIOSOptions ctor end" << std::endl;
#endif
    }
    
    //------------------------------------------------------------------------
    void set(const Node &opts)
    {
        if(opts.has_child("mpicomm"))
            mpicomm = opts["mpicomm"].as_int();

        // We need to initialize if we have not done so we can call
        // valid_transport and valid_transform.
        conduit_adios_initialize(mpicomm);

        if(opts.has_child("collective"))
            collective = opts["collective"].as_int() > 0;

        if(opts.has_child("buffer_size"))
            buffer_size = opts["buffer_size"].as_long();

        if(opts.has_child("transport"))
        {
            std::string s(opts["transport"].as_string());
            if(valid_transport(s))
                transport = s;
        }

        if(opts.has_child("statistics_flag"))
        {
            std::string s(opts["statistics_flag"].as_string());
            if(s == "adios_stat_no")
                statistics_flag = adios_stat_no;
            else if(s == "adios_stat_minmax")
                statistics_flag = adios_stat_minmax;
            else if(s == "adios_stat_default")
                statistics_flag = adios_stat_default;
            else if(s == "adios_stat_full")
                statistics_flag = adios_stat_full;
        }

        if(opts.has_child("transform"))
        {
            std::string s(opts["transform"].as_string());
            if(valid_transform(s))
                transform = s;
        }

#ifdef SEPARATE_OPTIONS
        if(opts.has_child("transports"))
        {
            const Node &n = opts["transports"];
            if(n.has_child("MPI")
            {
                const Node &tf = n["MPI"];
                if(tf.has_child("verbose"))
                    MPI_verbose = tf["verbose"].to_value();
            }
            else if(n.has_child("MPI_LUSTRE"))
            {
                const Node &tf = n["MPI_LUSTRE"];
                if(tf.has_child("stripe_count"))
                    MPI_LUSTRE_strip_count = tf["stripe_count"].to_value();
                if(tf.has_child("stripe_size"))
                    MPI_LUSTRE_strip_size = tf["stripe_size"].to_value();
                if(tf.has_child("block_count"))
                    MPI_LUSTRE_block_count = tf["block_count"].to_value();
            }
            else if(n.has_child("MPI_AGGREGATE"))
            {
                const Node &tf = n["MPI_AGGREGATE"];
                if(tf.has_child("num_aggregators"))
                    MPI_LUSTRE_num_aggregators = tf["num_aggregators"].to_value();
                if(tf.has_child("num_ost"))
                    MPI_LUSTRE_num_ost = tf["num_ost"].to_value();
                if(tf.has_child("have_metadata_file"))
                    MPI_LUSTRE_have_metadata_file = tf["have_metadata_file"].to_value();
                if(tf.has_child("striping"))
                    MPI_LUSTRE_striping = tf["striping"].to_value();
                if(tf.has_child("stripe_count"))
                    MPI_LUSTRE_stripe_count = tf["stripe_count"].to_value();
                if(tf.has_child("stripe_size"))
                    MPI_LUSTRE_stripe_size = tf["stripe_size"].to_value();
                if(tf.has_child("random_offset"))
                    MPI_LUSTRE_random_offset = tf["random_offset"].to_value();
                if(tf.has_child("local-fs"))
                    MPI_LUSTRE_localfs = tf["local-fs"].to_value();
            }
            else if(n.has_child("VAR_MERGE"))
            {
                const Node &tf = n["VAR_MERGE"];
                if(tf.has_child("chunk_size"))
                    MPI_LUSTRE_chunk_size = tf["chunk_size"].to_value();
                if(tf.has_child("io_method"))
                    MPI_LUSTRE_io_method = tf["io_method"].as_string();
                if(tf.has_child("io_parameters"))
                    MPI_LUSTRE_io_parameters = tf["io_parameters"].as_string();
                if(tf.has_child("num_aggregators"))
                    MPI_LUSTRE_num_aggregators = tf["num_aggregators"].to_value();
                if(tf.has_child("num_ost"))
                    MPI_LUSTRE_num_ost = tf["num_ost"].to_value();
            }
        }
#else
        if(opts.has_child("transport_options"))
            transport_options = opts["transport_options"].as_string();

        if(opts.has_child("transform_options"))
            transform_options = opts["transform_options"].as_string();
#endif

#if 0
        Node tmp;
        about(tmp);
        std::cout << "ADIOS options: " << tmp.to_json() << std::endl;
#endif
    }

    //------------------------------------------------------------------------
    void about(Node &opts)
    {
        opts.reset();

        opts["mpicomm"] = mpicomm;
        opts["collective"] = collective ? 1 : 0;
        opts["buffer_size"] = buffer_size;
        opts["transport"] = transport;
        if(statistics_flag == adios_stat_no)
            opts["statistics_flag"] = std::string("adios_stat_no");
        else if(statistics_flag == adios_stat_minmax)
            opts["statistics_flag"] = std::string("adios_stat_minmax");
        else if(statistics_flag == adios_stat_default)
            opts["statistics_flag"] = std::string("adios_stat_default");
        else if(statistics_flag == adios_stat_full)
            opts["statistics_flag"] = std::string("adios_stat_full");
        opts["transform"] = transform;

        // We need to initialize if we have not done so we can call some
        // ADIOS introspection functions.
        conduit_adios_initialize(mpicomm);

        std::vector<std::string> transports(conduit_adios_transports()),
                                 transforms(conduit_adios_transforms());
#ifdef SEPARATE_OPTIONS
        Node &tfopt = opts["transforms"];
        if(std::find(transports.begin(), transports.end(), "POSIX")
            != transforms.end())
        {
            tfopt["POSIX/local-fs"] = POSIX_localfs;
        }

        if(std::find(transports.begin(), transports.end(), "MPI")
            != transforms.end())
        {
            tfopt["MPI/verbose"] = MPI_verbose;
        }

        if(std::find(transports.begin(), transports.end(), "MPI_LUSTRE")
            != transforms.end())
        {
            tfopt["MPI_LUSTRE/stripe_count"] = MPI_LUSTRE_stripe_count;
            tfopt["MPI_LUSTRE/stripe_size"] = MPI_LUSTRE_stripe_size;
            tfopt["MPI_LUSTRE/block_count"] = MPI_LUSTRE_block_count;
        }

        if(std::find(transports.begin(), transports.end(), "MPI_AGGREGATE")
            != transforms.end())
        {
            tfopt["MPI_AGGREGATE/num_aggregators"] = MPI_AGGREGATE_num_aggregators;
            tfopt["MPI_AGGREGATE/num_ost"] = MPI_AGGREGATE_num_ost;
            tfopt["MPI_AGGREGATE/have_metadata_file"] = MPI_AGGREGATE_have_metadata_file;
            tfopt["MPI_AGGREGATE/striping"] = MPI_AGGREGATE_striping;
            tfopt["MPI_AGGREGATE/stripe_count"] = MPI_AGGREGATE_stripe_count;
            tfopt["MPI_AGGREGATE/stripe_size"] = MPI_AGGREGATE_stripe_size;
            tfopt["MPI_AGGREGATE/random_offset"] = MPI_AGGREGATE_random_offset;
            tfopt["MPI_AGGREGATE/local-fs"] = MPI_AGGREGATE_localfs;
        }

        if(std::find(transports.begin(), transports.end(), "VAR_MERGE")
            != transforms.end())
        {
            tfopt["VAR_MERGE/chunk_size"] = MPI_AGGREGATE_chunk_size;
            tfopt["VAR_MERGE/io_method"] = MPI_AGGREGATE_io_method;
            tfopt["VAR_MERGE/io_parameters"] = MPI_AGGREGATE_io_parameters;
            tfopt["VAR_MERGE/num_aggregators"] = MPI_AGGREGATE_num_aggregators;
            tfopt["VAR_MERGE/num_ost"] = MPI_AGGREGATE_num_ost;
        }
#else
        opts["transport_options"] = transport_options;
        opts["transform_options"] = transform_options;
#endif
        // Add in the available transports and transforms.
        std::string sep(", ");
        opts["read_only/available_transports"] = join_string_vector(transports, sep);
        opts["read_only/available_transforms"] = join_string_vector(transforms, sep);



/*
   adios/mpicomm = -1
   adios/buffer_size = 2^20
   adios/collective = true
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

// TODO: Let the user pass the MPI comm as an int

// TODO: inquire a list of transforms that could be applied to the data arrays (e.g. zlib)
//       adios_set_transform

// TODO: expose a "time_index" so the user can pass a time step.

//       This would let the ADIOS library compute statistics that might be useful.
//       Q: How to send them to Conduit logs???
// TODO: "statistics_flag" = "adios_stat_no";
//                           adios_stat_minmax, adios_stat_default, or adios_stat_full
    }

    //------------------------------------------------------------------------
    std::string GetTransportOptions() const
    {
#ifdef SEPARATE_OPTIONS
        std::ostringstream oss;

        if(transport == "POSIX")
        {
            oss << "local-fs=" << POSIX_localfs;
        }
        else if(transport == "MPI")
        {
            oss << "level=" << MPI_verbose;
        }
        else if(transport == "MPI_LUSTRE")
        {
            oss << "stripe_count=" << MPI_LUSTRE_stripe_count
                << ";stripe_size=" << MPI_LUSTRE_stripe_size
                << ";block_count=" << MPI_LUSTRE_block_count;
        }
        else if(transport == "MPI_AGGREGATE")
        {
            oss << "num_aggregators=" << MPI_AGGREGATE_num_aggregators
                << ";num_ost=" << MPI_AGGREGATE_num_ost
                << ";have_metadata_file=" << MPI_AGGREGATE_have_metadata_file
                << ";striping=" << MPI_AGGREGATE_striping
                << ";stripe_count=" << MPI_AGGREGATE_stripe_count
                << ";stripe_size=" << MPI_AGGREGATE_stripe_size
                << ";random_offset=" << MPI_AGGREGATE_random_offset
                << ";local-fs=" << MPI_AGGREGATE_localfs;
        }
        else if(transport == "VAR_MERGE")
        {
            oss << "num_aggregators=" << MPI_AGGREGATE_num_aggregators
                << ";num_ost=" << MPI_AGGREGATE_num_ost
                << ";have_metadata_file=" << MPI_AGGREGATE_have_metadata_file
                << ";striping=" << MPI_AGGREGATE_striping
                << ";stripe_count=" << MPI_AGGREGATE_stripe_count
                << ";stripe_size=" << MPI_AGGREGATE_stripe_size
                << ";random_offset=" << MPI_AGGREGATE_random_offset
                << ";local-fs=" << MPI_AGGREGATE_localfs;
        }
        return oss.str();
#else
        return transport_options;
#endif
    }

    //------------------------------------------------------------------------
    std::string GetTransformOptions() const
    {
         return transform_options;
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
std::cout << "GetOptions: start" << std::endl;
    if(adiosState_options == NULL)
    {
std::cout << "GetOptions: creating new options" << std::endl;
        adiosState_options = new ADIOSOptions;
        atexit(CleanupOptions);
    }
std::cout << "GetOptions: end" << std::endl;
    return adiosState_options;
}

//-----------------------------------------------------------------------------
static void iterate_conduit_node(const conduit::Node &node,
    void (*func)(const conduit::Node &, void *), void *funcData)
{
    if(node.number_of_children() == 0)
    {
        func(node, funcData);
    }
    else
    {
        for(conduit::index_t i = 0; i < node.number_of_children(); ++i)
            iterate_conduit_node(node.child(i), func, funcData);
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
            if(node.dtype().is_integer())
                dtype = adios_integer;
            else if(node.dtype().is_signed_integer())
                dtype = adios_integer;
            else if(node.dtype().is_unsigned_integer())
                dtype = adios_unsigned_integer;
            else if(node.dtype().is_int8())
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
            dtype = adios_string;
        }
    }

    return dtype;
}

static void define_variables(const Node &node, void *funcData)
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

    // Dimensions
    char dimensions[20];
    const char *global_dimensions = "";
    const char *local_offsets = "";
    if(node.dtype().number_of_elements() > 1)
        sprintf(dimensions, "%ld", node.dtype().number_of_elements());
    else
        dimensions[0] = '\0';

    // Define the variable.
    int64_t vid = adios_define_var(state->gid, node.path().c_str(), 
                      "", dtype,
                      dimensions, global_dimensions, local_offsets);
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
        std::string transform(GetOptions()->transform);
        if(!GetOptions()->transform_options.empty())
        {
            transform += ":";
            transform += GetOptions()->transform_options;
        }
        if(adios_set_transform(vid, transform.c_str()) != 0)
        {
            CONDUIT_ERROR("ADIOS Error:" << adios_get_last_errmsg()); 
        }
    }

    // Store the name of the actual Conduit type as an attribute in case we 
    // need it to read.
    if(adios_define_attribute(state->gid, "conduit_type", "", adios_string, 
        node.dtype().name().c_str(), node.path().c_str()) < 0)
    {
        CONDUIT_ERROR("ADIOS Error:" << adios_get_last_errmsg()); 
        return;
    }

    // Add the variable's var size to the total for the group.
    state->gSize += adios_expected_var_size(vid);
}

//-----------------------------------------------------------------------------
static void write_variables(const Node &node, void *funcData)
{
    adios_save_state *state = (adios_save_state *)funcData;

    // Map Conduit types to ADIOS types.
    ADIOS_DATATYPES dtype = conduit_dtype_to_adios_dtype(node);
    if(dtype == adios_unknown)
    {
        CONDUIT_ERROR("Unsupported Conduit to ADIOS type conversion.");
        return;
    }

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
static bool conduit_adios_declare_group(int64_t *gid)
{
    //
    // Group
    //
    const char *time_index = "";
    // TODO: See if there is a time index in the ADIOS options?
    // TODO: Do we need to declare the group if we are just adding a time step?
    if(adios_declare_group(gid, "conduit", time_index, 
        GetOptions()->statistics_flag) != 0)
    {
        CONDUIT_ERROR("ADIOS error: " << adios_get_last_errmsg());
        return false;
    }

    std::string transport(GetOptions()->transport);
    std::string parameters(GetOptions()->GetTransportOptions());
    const char *base_path = ""; // blank for current directory.
    if(adios_select_method(*gid,
                           transport.c_str(),
                           parameters.c_str(),
                           base_path))
    {
        CONDUIT_ERROR("ADIOS error: " << adios_get_last_errmsg());
        return false;
    }

    return true;
}

//-----------------------------------------------------------------------------
static int conduit_adios_save(const Node &node, const std::string &path, 
    adios_save_state *state, const char *flag, MPI_Comm comm)
{
    // NOTE: We have already done the group and variable declarations.
    // NOTE: Assume the path will be the same on all ranks in the comm.

    //
    // Open the file
    //
    if(adios_open(&state->fid, "conduit", path.c_str(), flag, comm) != 0)
    {
        CONDUIT_ERROR("ADIOS error: " << adios_get_last_errmsg());
        return -1;
    }

    // This is an optional call that lets ADIOS size its output buffers.
    uint64_t total_size = 0;
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
    iterate_conduit_node(node, write_variables, state);

    //
    // Close the file.
    //
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
std::cout << "adios_options: Calling GetOptions" << std::endl;
    GetOptions()->about(opts);
}

//-----------------------------------------------------------------------------
void
adios_save(const Node &node, const std::string &path)
{
    std::string filename(path);

    //
    // MPI Communicator, filename
    //
#ifdef _NOMPI
    conduit_adios_initialize(GetOptions()->mpicomm);
#else
    int oldComm = GetOptions()->mpicomm;
    MPI_Comm comm;
    if(!GetOptions()->collective)
    {
        // Split the comm so each rank is separate since we're not 
        // running collectively. Poke the new comm into the options
        // and we'll restore it.
        int rank = 0;
        MPI_Comm_rank(MPI_Comm_f2c(GetOptions()->mpicomm), &rank);
        MPI_Comm_split(MPI_Comm_f2c(GetOptions()->mpicomm), rank, 0, &comm);
        GetOptions()->mpicomm = MPI_Comm_c2f(comm);
    }
    else
    {
        comm = MPI_Comm_f2c(GetOptions()->mpicomm);
        int rank = 0;
        MPI_Comm_rank(comm, &rank);

        // Bcast the filename so we know it will be the same on all ranks.
        int len = path.size()+1;
        MPI_Bcast(&len, 1, MPI_INT, 0, comm);
        char *sbuf = new char[len];
        if(rank == 0)
            strcpy(sbuf, path.c_str());
        MPI_Bcast(sbuf, len, MPI_CHAR, 0, comm);
        if(rank > 0)
            filename = std::string(sbuf);
        delete [] sbuf;
    }

    // Initialize ADIOS using the new comm.
    conduit_adios_initialize(GetOptions()->mpicomm);
#endif

    // Set ADIOS's max buffer sized based on the options.
    adios_set_max_buffer_size(static_cast<uint64_t>(GetOptions()->buffer_size));

    adios_save_state state;
    state.fid = 0;
    state.gid = 0;
    state.gSize = 0;

    //
    // Group
    //
    if(conduit_adios_declare_group(&state.gid) == 0)
    {
        //
        // Define Variables
        //
        iterate_conduit_node(node, define_variables, &state);

        //
        // Save the data.
        //
#ifndef _NOMPI
        conduit_adios_save(node, filename, &state, "w", 0);
#else
        conduit_adios_save(node, filename, &state, "w", comm);
#endif
    }

#ifndef _NOMPI
    // If we had split the comm due to non-collective comm. Free that comm.
    if(oldComm != GetOptions()->mpicomm)
    {
        MPI_Comm_free(&comm);
        GetOptions()->mpicomm = oldComm;
    }
#endif
}

//-----------------------------------------------------------------------------
void adios_append(const Node &node, const std::string &path)
{
    std::cout << "conduit::relay::io::adios_append(node, path=" << path << ")" << std::endl;
}

//-----------------------------------------------------------------------------
void adios_load(const std::string &path, Node &node)
{
    std::cout << "conduit::relay::io::adios_load(node, path=" << path << ")" << std::endl;

// NOTE: If we're loading a dataset and we give it a path, we want to load 
//       just this processor's piece...
//       For reading, that's not generally true as we might want to use a serial
//       read library to read separate domains that were all saved to the same file
//       and we'd want control over which parts we read out.

}


}
//-----------------------------------------------------------------------------
// -- end conduit::relay::io --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit::relay --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------
