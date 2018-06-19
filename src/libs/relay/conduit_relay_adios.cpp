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
    std::cout << "conduit_adios_initialize: init'd=" << adiosState_initialized << std::endl;
    if(!adiosState_initialized)
    {
#ifdef USE_MPI
        // See if MPI is initialized.
        int mpi_init = 0;
        MPI_Initialized(&mpi_init);
        std::cout << "mpi_init = " << mpi_init << std::endl;

        // Initialize ADIOS.
        int status = adios_init_noxml(comm);
        std::cout << "adios_init_noxml = " << status << std::endl;
#else
        std::cout << "initializing serial ADIOS." << std::endl;
        // Initialize ADIOS.
        int status = adios_init_noxml(comm);
        std::cout << "adios_init_noxml = " << status << std::endl;
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
    ADIOS_READ_METHOD read_method;
    std::string       read_parameters;
    ADIOS_LOCKMODE    read_lock_mode;
    int               read_verbose;
    float             read_timeout;
public:
    ADIOSOptions() :  
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
        VAR_MERGE_num_ost(672),
#else
        transport_options(),
        transform_options(),
#endif
        // Read options
        read_method(ADIOS_READ_METHOD_BP),
        read_parameters(),
        read_lock_mode(ADIOS_LOCKMODE_CURRENT),
        read_verbose(0),
        read_timeout(0.f)
    {
std::cout << "ADIOSOptions ctor start" << std::endl;
#ifdef USE_MPI
        int mpi_init = 0;
        MPI_Initialized(&mpi_init);
        std::cout << "mpi_init = " << mpi_init << std::endl;


        transport = "MPI";
std::cout << "ADIOSOptions ctor end" << std::endl;
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

        // Read options
        if(opts.has_child("read"))
        {
            const Node &n = opts["read"];
            if(n.has_child("read_method"))
            {
                std::string s(n["read_method"].as_string());
                if(s == "ADIOS_READ_METHOD_BP")
                    read_method = ADIOS_READ_METHOD_BP;
                else if(s == "ADIOS_READ_METHOD_BP_AGGREGATE")
                    read_method = ADIOS_READ_METHOD_BP_AGGREGATE;
                else if(s == "ADIOS_READ_METHOD_DATASPACES")
                    read_method = ADIOS_READ_METHOD_DATASPACES;
                else if(s == "ADIOS_READ_METHOD_DIMES")
                    read_method = ADIOS_READ_METHOD_DIMES;
                else if(s == "ADIOS_READ_METHOD_FLEXPATH")
                    read_method = ADIOS_READ_METHOD_FLEXPATH;
            }

            if(n.has_child("parameters"))
                read_parameters = n["parameters"].as_string();

            if(n.has_child("lock_mode"))
            {
                std::string s(n["lock_mode"].as_string());
                if(s == "ADIOS_LOCKMODE_NONE")
                    read_lock_mode = ADIOS_LOCKMODE_NONE;
                else if(s == "ADIOS_LOCKMODE_CURRENT")
                    read_lock_mode = ADIOS_LOCKMODE_CURRENT;
                else if(s == "ADIOS_LOCKMODE_ALL")
                    read_lock_mode = ADIOS_LOCKMODE_ALL;
            }

            if(n.has_child("timeout"))
                read_timeout = n["timeout"].as_float();

            if(n.has_child("verbose"))
                read_verbose = n["verbose"].to_value();
        }

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
#ifdef USE_MPI
        conduit_adios_initialize(MPI_COMM_WORLD);
#else
        conduit_adios_initialize(0);
#endif

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
    if(adiosState_options == NULL)
    {
        adiosState_options = new ADIOSOptions;
        atexit(CleanupOptions);
    }
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
    if(dtype != adios_string && node.dtype().number_of_elements() > 1)
        sprintf(dimensions, "%ld", node.dtype().number_of_elements());
    else
        dimensions[0] = '\0';

    // Define the variable.
std::cout << "adios_define_var: " << node.path() << ", dtype=" << dtype << ", dims=" << dimensions << std::endl;
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
std::cout << "adios_set_transform " << transform << std::endl;
        if(adios_set_transform(vid, transform.c_str()) != 0)
        {
            CONDUIT_ERROR("ADIOS Error:" << adios_get_last_errmsg()); 
        }
    }

    // Store the name of the actual Conduit type as an attribute in case we 
    // need it to read.
    if(adios_define_attribute(state->gid, "conduit_type", node.path().c_str(), adios_string, 
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
std::cout << "adios_write: " << node.path() << std::endl;
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
std::cout << "adios_declare_group: conduit" << std::endl;
    int retval = adios_declare_group(gid, "conduit", time_index, 
        GetOptions()->statistics_flag);
std::cout << "adios_declare_group: returned " << retval << std::endl;
    if(retval != 0)
    {
        CONDUIT_ERROR("ADIOS error: " << adios_get_last_errmsg());
        return false;
    }

    std::string transport(GetOptions()->transport);
    std::string parameters(GetOptions()->GetTransportOptions());
    const char *base_path = ""; // blank for current directory.
std::cout << "adios_select_method: transport = "<< transport << std::endl;
    retval = adios_select_method(*gid,
                           transport.c_str(),
                           parameters.c_str(),
                           base_path);
std::cout << "adios_select_method: returned "<< retval << std::endl;
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
    std::cout << "adios_open: path=" << path << ", flag=" << flag << std::endl;
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
    std::cout << "adios_close" << std::endl;
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
adios_save(const Node &node, const std::string &path
   CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm comm)
   )
{
    std::string filename(path);

#ifdef USE_MPI
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

    std::cout << rank << ": filename=" << filename << std::endl;

    // Initialize ADIOS.
    conduit_adios_initialize(comm);
#else
    // Initialize ADIOS.
    conduit_adios_initialize(0);
#endif

    // Set ADIOS's max buffer sized based on the options.
    adios_set_max_buffer_size(static_cast<uint64_t>(GetOptions()->buffer_size));
std::cout << "buffer_size = " << GetOptions()->buffer_size << std::endl;
    adios_save_state state;
    state.fid = 0;
    state.gid = 0;
    state.gSize = 0;

    //
    // Group
    //
    if(conduit_adios_declare_group(&state.gid))
    {
std::cout << "created group." << std::endl;
        //
        // Define Variables
        //
        iterate_conduit_node(node, define_variables, &state);

        //
        // Save the data.
        //
#ifdef USE_MPI
        conduit_adios_save(node, filename, &state, "w", comm);
#else
std::cout << "Callng serial conduit_adios_save" << std::endl;
        conduit_adios_save(node, filename, &state, "w", 0);
#endif
    }
    else
    {
std::cout << "failed to create group." << std::endl;
    }

    // Delete the variable definitions from the group so we can define them
    // again the next time around. Free the group too.
std::cout << "adios_delete_vardefs" << std::endl;
    adios_delete_vardefs(state.gid);
std::cout << "adios_free_group" << std::endl;
    adios_free_group(state.gid);
}

//-----------------------------------------------------------------------------
void adios_append(const Node &node, const std::string &path
   CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm comm)
   )
{
    std::cout << "conduit::relay::io::adios_append(node, path=" << path << ")" << std::endl;
}

//-----------------------------------------------------------------------------
bool name_matches_subtree(const std::string &name, const std::string &pattern)
{
    // for now.
    return true;
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

int rank = 0;
#ifdef USE_MPI
    MPI_Comm_rank(comm, &rank);
#endif
    std::cout << rank << ": adios_read_init_method()" << std::endl;

    // once per program run...
    adios_read_init_method(GetOptions()->read_method, 
#ifdef USE_MPI
                           comm,
#else
                           0,
#endif
                           GetOptions()->read_parameters.c_str()
                           );
    std::cout << rank << ": adios_read_open(" << path << ")" << std::endl;

    // Open the file for read.
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

            // Test that the variable is something we want to read.
            if(!name_matches_subtree(vname, pattern))
                continue;

            ADIOS_VARINFO *v = adios_inq_var(afile, afile->var_namelist[i]);
            if(v)
            {
#if 0
                if(v->ndim == 0)
                {
// NOTE: if multiple ranks wrote out their piece, how can the data be in the v object?
//       Do we need a selection applied here already?


                    adios_inq_var_blockinfo(afile,v);

                    // The number of blocks in the current step.
                    int nblocks = v->nblocks[ts];

std::cout << "scalar " << vname << " has " << nblocks << " blocks" << std::endl;

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
                    std::cout << rank << ": " << vname <<" = " << n.to_json() << std::endl;
#endif
                }
                else
                {
#endif
                    // We have array data. Let's allocate a buffer for 
                    // it and schedule a read.

                    adios_inq_var_blockinfo(afile,v);

                    // The number of blocks in the current step.
                    int nblocks = v->nblocks[ts];

// TODO: make sure which_block is in range...
    std::cout << rank << ": adios_selection_writeblock(" << domain
              << ")  (nblocks=" << nblocks << ")" << std::endl;

#if 1
if(rank == 1)
{
    std::cout << "vname=" << vname << std::endl << "{" << std::endl;
    std::cout << "   type = " << v->type << std::endl;
    std::cout << "   ndim = " << v->ndim << std::endl;
    std::cout << "   dims = {";
    for(int q = 0; q < v->ndim; ++q)
        std::cout << "," << v->dims[q];
    std::cout << "}" << std::endl;
    std::cout << "   nsteps = " << v->nsteps << std::endl;
    std::cout << "   global = " << v->global << std::endl;
    std::cout << "   nblocks = {";
    for(int q = 0; q < v->nsteps; ++q)
        std::cout << "," << v->nblocks[q];
    std::cout << "}" << std::endl;

    std::cout << "   blockinfo = {" << std::endl;
    for(int bi = 0; bi < v->sum_nblocks; ++bi)
    {
        std::cout << "       {" << std::endl;

        std::cout << "           start={";
        for(int q = 0; q < v->ndim; ++q)
            std::cout << "," << v->blockinfo[bi].start[q];
        std::cout << "}" << std::endl;

        std::cout << "           count={";
        for(int q = 0; q < v->ndim; ++q)
            std::cout << "," << v->blockinfo[bi].count[q];
        std::cout << "}" << std::endl;

        std::cout << "           process_id=" << v->blockinfo[bi].process_id << std::endl;
        std::cout << "           time_index=" << v->blockinfo[bi].time_index << std::endl;
        std::cout << "       }," << std::endl;
    }
    std::cout << "   }" << std::endl;

    std::cout << "}" << std::endl;
}
#endif


                    // Select the block we want.
                    ADIOS_SELECTION *sel = adios_selection_writeblock(domain);
                    sels.push_back(sel);

#if 0
                    // Use the dims to figure out the number of elements.
                    uint64_t nelem = 1;
                    for(int d = 0; d < v->ndim; ++d)
                        nelem *= std::max(uint64_t(1), v->dims[d]);
#else
                    // We can't rely on the v->dims being good for anything but the first
                    // MPI rank, at least the way we're writing the data. Compute the
                    // number of elements from the blockinfo so we can properly size the
                    // destination array.

                    // NOTE: can we assume these will be in process order?
                    int ts1 = ts + 1;
                    uint64_t nelem = 1;
                    for(int bi = 0; bi < v->sum_nblocks; ++bi)
                    {
                        if(v->blockinfo[bi].time_index == ts1 &&
                           v->blockinfo[bi].process_id == domain)
                        {
                            nelem = v->blockinfo[bi].count[0];
                            break;
                        }
                    }
#endif
std::cout << rank << ": " << vname << " nelem=" << nelem << std::endl;

                    // Allocate memory for the variable.
                    void *vbuf = NULL;
                    if(v->type == adios_byte)
                    {
                        node[vname].set(DataType::int8(nelem));
                        int8 *buf = node[vname].value();
                        vbuf = (void *)buf;
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
                    else if(v->type == adios_string)
                    {
std::cout << rank << ": adios_string: nelem=" << nelem << std::endl;

int slen = strlen((char *)v->value);
nelem = slen;
std::cout << rank << ": adios_string: slen=" << slen << " str=" << ((char *)v->value) << std::endl;

                        char *buf = new char[nelem + 1];
                        memset(buf, 0, (nelem+1)*sizeof(char));
                        node[vname].set(buf);
                        vbuf = (void *)buf;
                    }
                    else
                    {
                        // Other cases should not happen.
                        CONDUIT_ERROR("Unsupported type " << v->type);
                    }

                    // Schedule the read.
                    if(vbuf != NULL)
                    {
    std::cout << rank << ": adios_schedule_read_byid(" << vname << ")" << std::endl;
                        adios_schedule_read_byid(afile, sel, v->varid, ts, 1, vbuf);
                        scheduled_reads++;
                    }
#if 0
                }
#endif

                adios_free_varinfo(v);
            }           
        }
#ifdef USE_MPI
MPI_Barrier(comm);
#endif

        // Perform any outstanding reads, blocking until reads are done.
        if(scheduled_reads > 0)
        {
    std::cout << rank << ": adios_perform_reads()" << std::endl;
            int blocking = 1;
            int ret = adios_perform_reads(afile, blocking);
        }
#if 0
        // Free the selections.
        for(size_t s = 0; s < sels.size(); ++s)
            adios_selection_delete(sels[s]);
#endif

    std::cout << rank << ": adios_read_close()" << std::endl;

        // Close the file.
        adios_read_close(afile);
    }
    else
    {
        CONDUIT_ERROR("ADIOS Error: " << adios_get_last_errmsg());
    }
    std::cout << rank << ": adios_read_finalize_method()" << std::endl;

    // once per program run.
    adios_read_finalize_method(GetOptions()->read_method);
}

//-----------------------------------------------------------------------------
void adios_load(const std::string &path, Node &node
   CONDUIT_RELAY_COMMUNICATOR_ARG(MPI_Comm comm)
   )
{
    int time_step = -1; // use current time step.
#ifdef USE_MPI
    // Read the rank'th domain if there is one.
    int domain = 0;
    MPI_Comm_rank(comm, &domain);
    adios_load(path, node, time_step, domain, comm);
#else
    int domain = 0;     // use first domain.
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
