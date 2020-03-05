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
/// file: conduit_blueprint_sarray.cpp
///
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit_blueprint_sarray.hpp"
#include "conduit_log.hpp"

//-----------------------------------------------------------------------------
// -- standard cpp lib includes -- 
//-----------------------------------------------------------------------------
#include <algorithm>
#include <map>
#include <set>
#include <limits>

using namespace conduit;
// Easier access to the Conduit logging functions
using namespace conduit::utils;

//-----------------------------------------------------------------------------
// -- begin conduit:: --
//-----------------------------------------------------------------------------
namespace conduit
{


//-----------------------------------------------------------------------------
// -- begin conduit::blueprint:: --
//-----------------------------------------------------------------------------
namespace blueprint
{

//-----------------------------------------------------------------------------
// - begin internal helper functions -
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
bool verify_field_exists(const std::string &protocol,
	const conduit::Node &node,
	conduit::Node &info,
	const std::string &field_name = "")
{
	bool res = true;

	if (field_name != "")
	{
		if (!node.has_child(field_name))
		{
			log::error(info, protocol, "missing child" + log::quote(field_name, 1));
			res = false;
		}

		log::validation(info[field_name], res);
	}

	return res;
}

//-----------------------------------------------------------------------------
bool verify_integer_field(const std::string &protocol,
	const conduit::Node &node,
	conduit::Node &info,
	const std::string &field_name = "")
{
	Node &field_info = (field_name != "") ? info[field_name] : info;

	bool res = verify_field_exists(protocol, node, info, field_name);
	if (res)
	{
		const Node &field_node = (field_name != "") ? node[field_name] : node;

		if (!field_node.dtype().is_integer())
		{
			log::error(info, protocol, log::quote(field_name) + "is not an integer (array)");
			res = false;
		}
	}

	log::validation(field_info, res);

	return res;
}

//-----------------------------------------------------------------------------
bool verify_number_field(const std::string &protocol,
	const conduit::Node &node,
	conduit::Node &info,
	const std::string &field_name = "")
{
	Node &field_info = (field_name != "") ? info[field_name] : info;

	bool res = verify_field_exists(protocol, node, info, field_name);
	if (res)
	{
		const Node &field_node = (field_name != "") ? node[field_name] : node;

		if (!field_node.dtype().is_number())
		{
			log::error(info, protocol, log::quote(field_name) + "is not a number");
			res = false;
		}
	}

	log::validation(field_info, res);

	return res;
}


//-----------------------------------------------------------------------------
bool verify_object_field(const std::string &protocol,
	const conduit::Node &node,
	conduit::Node &info,
	const std::string &field_name = "",
	const bool allow_list = false,
	const index_t num_children = 0)
{
	Node &field_info = (field_name != "") ? info[field_name] : info;

	bool res = verify_field_exists(protocol, node, info, field_name);
	if (res)
	{
		const Node &field_node = (field_name != "") ? node[field_name] : node;

		if (!(field_node.dtype().is_object() ||
			(allow_list && field_node.dtype().is_list())))
		{
			log::error(info, protocol, log::quote(field_name) + "is not an object" +
				(allow_list ? " or a list" : ""));
			res = false;
		}
		else if (field_node.number_of_children() == 0)
		{
			log::error(info, protocol, "has no children");
			res = false;
		}
		else if (num_children && field_node.number_of_children() != num_children)
		{
			std::ostringstream oss;
			oss << "has incorrect number of children ("
				<< field_node.number_of_children()
				<< " vs "
				<< num_children
				<< ")";
			log::error(info, protocol, oss.str());
			res = false;
		}
	}

	log::validation(field_info, res);

	return res;
}


//-----------------------------------------------------------------------------
// -- end internal helper functions --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::sarray --
//-----------------------------------------------------------------------------
namespace sarray
{

//-----------------------------------------------------------------------------
bool
verify(const std::string &/*protocol*/,
       const Node &/*n*/,
       Node &info)
{
    // sarray doesn't provide any nested protocols
  
    info.reset();
    log::validation(info,false);
    return false;
}

//----------------------------------------------------------------------------
bool
verify(const conduit::Node &sarray,
       Node &info)
{
    info.reset();
    bool res = true;

    const std::string protocol = "sarray";

    // sarray needs to be an object
    if( ! (sarray.dtype().is_object() ) )
    {
        log::error(info,protocol,"Node has no children");
        res = false;
    }

    res &= verify_integer_field(protocol, sarray, info, "n");

    res &= verify_number_field(protocol, sarray, info, "nz");

    res &= verify_field_exists(protocol, sarray, info, "idx");
    if(res)
    {
        const Node &idx = sarray["idx"];
        int elements = -1;

        // idx can be an integer array
        if (!idx.dtype().is_list())
        {
            res &= verify_integer_field(protocol, sarray, info, "idx");
            elements = idx.dtype().number_of_elements();
        }
        // idx can also be a list of arrays
        else
        {
            NodeConstIterator itr = idx.children();
            while(itr.has_next())
            {
                const Node& theidx = itr.next();
                res &= verify_integer_field(protocol, theidx, info);
                if (elements == -1)
                {
                    elements = theidx.dtype().number_of_elements();
                }
            }
        }

        // idx[0] must have the same number of elements as nz
        if (elements != sarray["nz"].dtype().number_of_elements())
        {
            log::error(info, protocol, "different number of elements in idx and nz");
            res = false;
        }
    }

	log::validation(info, res);
    
    return res;
}


//-----------------------------------------------------------------------------
int
nnz(const conduit::Node &n)
{
    return n["nz"].dtype().number_of_elements();
}

//-----------------------------------------------------------------------------
int
length(const conduit::Node &n)
{
    return n["n"].as_int64();
}

//-----------------------------------------------------------------------------
void
make_sparse(const conduit::Node &src,
            conduit::Node dst)
{
    // TODO (AGC): make a sparse representation dst from src
}

//-----------------------------------------------------------------------------
void
make_full(const conduit::Node &src,
          conduit::Node dst)
{
    // TODO (AGC): make a full (zeros included) representation dst from src
}
}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::sarray --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint:: --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------
