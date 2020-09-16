// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_relay_zfp.cpp
///
//-----------------------------------------------------------------------------

#include "conduit_relay_zfp.hpp"
#include "zfpfactory.h"

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

zfp::array*
unwrap_zfparray(const Node &node)
{
    Node compressed_data = node.fetch_existing(ZFP_COMPRESSED_DATA_FIELD_NAME);

    // verify word size is readable with zfp
    // zfp's bitstream consists of uint words
    bool is_readable = true;
    switch(stream_word_bits) {
        case 64:
            is_readable = compressed_data.dtype().is_uint64();
            break;

        case 32:
            is_readable = compressed_data.dtype().is_uint32();
            break;

        case 16:
            is_readable = compressed_data.dtype().is_uint16();
            break;

        case 8:
            is_readable = compressed_data.dtype().is_uint8();
            break;

        default:
            is_readable = false;
            break;
    }

    if(!is_readable) {
        return NULL;
    }

    zfp::array::header header;
    memcpy(header.buffer, node.fetch_existing(ZFP_HEADER_FIELD_NAME).data_ptr(), sizeof(header));

    try {
        return zfp::array::construct(header, static_cast<uchar*>(compressed_data.data_ptr()), compressed_data.allocated_bytes());
    } catch(std::exception const &) {
        // could be zfp::array::header::exception, or std::bad_alloc
        return NULL;
    }
}

template<typename T>
void
cast_and_set_compressed_data(Node &dest, uchar* compressed_data, size_t num_data_words)
{
    void* intermediate_ptr = static_cast<void*>(compressed_data);
    dest[ZFP_COMPRESSED_DATA_FIELD_NAME].set(static_cast<T*>(intermediate_ptr), num_data_words);
}

int
wrap_zfparray(const zfp::array *arr,
         Node &dest)
{
    // store header
    zfp::array::header header;
    try {
        header = arr->get_header();
    } catch(zfp::array::header::exception const &) {
        return 1;
    }
    dest[ZFP_HEADER_FIELD_NAME].set(static_cast<uint8*>(header.buffer), sizeof(header));

    // store compressed data
    size_t compressed_data_len_bits = arr->compressed_size() * CHAR_BIT;
    // should already by a multiple of stream_word_bits (round for safety)
    size_t num_data_words = (compressed_data_len_bits + stream_word_bits - 1) / stream_word_bits;

    // store compressed-data under same dtype as bitstream's underlying word
    switch(stream_word_bits) {
        case 64:
            cast_and_set_compressed_data<uint64>(dest, arr->compressed_data(), num_data_words);
            break;

        case 32:
            cast_and_set_compressed_data<uint32>(dest, arr->compressed_data(), num_data_words);
            break;

        case 16:
            cast_and_set_compressed_data<uint16>(dest, arr->compressed_data(), num_data_words);
            break;

        case 8:
            cast_and_set_compressed_data<uint8>(dest, arr->compressed_data(), num_data_words);
            break;

        default:
            // error
            return 2;
    }

    return 0;
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
