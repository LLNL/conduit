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
zfp_write(const Node &node)
{
    zfp::array::header header[1];
    memcpy(header->buffer, node.fetch_child(ZFP_HEADER_FIELD_NAME).data_ptr(), sizeof(header));

    Node compressedData = node.fetch_child(ZFP_COMPRESSED_DATA_FIELD_NAME);

    try {
        return zfp::array::construct(header[0], static_cast<uchar*>(compressedData.data_ptr()), compressedData.allocated_bytes());
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
zfp_read(const zfp::array *arr,
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
            return 1;
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
