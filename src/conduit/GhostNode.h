#ifndef conduit_ghostnode_h__
#define conduit_ghostnode_h__


#include"Node.h"
namespace conduit {

   class GhostNode {
      public:
         GhostNode(void *ptr, Node const &n);
         GhostNode(GhostNode const &a);
         ~GhostNode();
         
         GhostNode& operator=(GhostNode const &a);
         
         template<typename T>
         T getpp(void){
            return m_node->getpp<T>(m_data);
         }
         
         GhostNode operator[](const std::string &path)
                      {return GhostNode(m_data, const_cast<Node*>(m_node)->fetch(path));}
    
         
      private:
         void *m_data;
         Node const *m_node;
   };

}


#endif

