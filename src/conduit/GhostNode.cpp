#include "GhostNode.h"

namespace conduit
{

GhostNode::GhostNode(void *ptr, Node const &n) :
   m_data(ptr),
   m_node(&n)
{
}

GhostNode::GhostNode(GhostNode const &a) :
   m_data(a.m_data),
   m_node(a.m_node)
{
}

GhostNode::~GhostNode()
{
}
 
 
GhostNode &
GhostNode::operator=(GhostNode const &a){
   m_data = a.m_data;
   m_node = a.m_node;
   return *this;
}  
   
}
