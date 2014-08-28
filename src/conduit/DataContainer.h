/*****************************************************************************
* Copyright (c) 2014, Lawrence Livermore National Security, LLC
* Produced at the Lawrence Livermore National Laboratory. 
* 
* All rights reserved.
* 
* This source code cannot be distributed without further review from 
* Lawrence Livermore National Laboratory.
*****************************************************************************/

///
/// file: DataContainer.h
///

#ifndef __CONDUIT_DATA_CONTAINER_H
#define __CONDUIT_DATA_CONTAINER_H

#include "Core.h"
#include "Error.h"
#include "Endianness.h"
#include "DataType.h"
#include "DataArray.h"
#include "Schema.h"

#include <map>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>


namespace conduit
{

class DataContainer
{
public:    
    
    /* Constructors */
              DataContainer(); 
              DataContainer(const DataContainer &container);

    /* Destructor */
    virtual  ~DataContainer();

    /* Allocation */
    void      allocate(index_t dsize); 
    void      allocate(const DataType &dtype); 
    void      mmap(const std::string &stream_path,index_t dsize);

    /* Cleanup */
    void      release();

    /* Access */
    void     *data_ptr() const {return m_data;}
    
private:

    void      cleanup();
    void      init_defaults();
    
    void     *m_data;
    bool      m_alloced;
    bool      m_mmaped;
    int       m_mmap_fd;
    index_t   m_mmap_size;

};

}


#endif
