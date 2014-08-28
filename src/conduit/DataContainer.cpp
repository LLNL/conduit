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
/// file: DataContainer.cpp
///

#include "DataContainer.h"
#include <iostream>
#include <cstdio>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace conduit
{

//============================================
/// DataContainer
//============================================
void
DataContainer::init_defaults()
{
    m_data = NULL;
    m_alloced = false;

    m_mmaped    = false;
    m_mmap_fd   = -1;
    m_mmap_size = 0;  
}

//============================================
DataContainer::DataContainer()
{
    init_defaults();
}

//============================================
DataContainer::DataContainer(const DataContainer &container)
{
    init_defaults();
    // we don't copy or transfer ownership in this case, 
    // it is a shallow copy
    m_data = container.data_ptr();
}

//============================================
DataContainer::~DataContainer()
{
    cleanup();
}

//============================================
void
DataContainer::release()
{
    cleanup();
    init_defaults();
}

//============================================
void
DataContainer::allocate(const DataType &dtype)
{
    // TODO: This implies compact storage
    allocate(dtype.number_of_elements()*dtype.element_bytes());
}

//============================================
void
DataContainer::allocate(index_t dsize)
{
    release();
    m_data    = malloc(dsize);
    m_alloced = true;
    m_mmaped  = false;
}


//============================================
void
DataContainer::mmap(const std::string &stream_path, index_t dsize)
{   
    release();
    m_mmap_fd   = open(stream_path.c_str(),O_RDWR| O_CREAT);
    m_mmap_size = dsize;

    if (m_mmap_fd == -1) 
        THROW_ERROR("<DataContainer::mmap> failed to open: " << stream_path);

    m_data = ::mmap(0, dsize, PROT_READ | PROT_WRITE, MAP_SHARED, m_mmap_fd, 0);

    if (m_data == MAP_FAILED) 
        THROW_ERROR("<DataContainer::mmap> MAP_FAILED" << stream_path);
    
    m_alloced = false;
    m_mmaped  = true;
}


//============================================
void
DataContainer::cleanup()
{
    if(m_alloced && m_data)
    {
        // clean up our storage
        free(m_data);
        m_data = NULL;
        m_alloced = false;
    }   
    else if(m_mmaped && m_data)
    {
        if(munmap(m_data, m_mmap_size) == -1) 
        {
            // error
        }
        close(m_mmap_fd);
        m_data      = NULL;
        m_mmap_fd   = -1;
        m_mmap_size = 0;
    }
}



};


