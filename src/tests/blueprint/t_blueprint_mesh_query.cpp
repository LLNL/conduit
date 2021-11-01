// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: t_blueprint_mesh_query.cpp
///
//-----------------------------------------------------------------------------

#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_log.hpp"

#include <set>
#include <vector>
#include <string>
#include "gtest/gtest.h"

using namespace conduit;
using namespace conduit::utils;

/// Testing Constants ///

/// Testing Helpers ///

/// Test Cases ///

//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_query, mesh_domains)
{
    { // Empty Tests //
        { // Empty Node Test //
            Node mesh;
            const std::vector<Node *> domains = blueprint::mesh::domains(mesh);
            ASSERT_EQ(domains.size(), 0);
        }

        { // Empty Object Test //
            Node mesh(DataType::object());
            const std::vector<Node *> domains = blueprint::mesh::domains(mesh);
            ASSERT_EQ(domains.size(), 0);
        }

        { // Empty List Test //
            Node mesh(DataType::list());
            const std::vector<Node *> domains = blueprint::mesh::domains(mesh);
            ASSERT_EQ(domains.size(), 0);
        }
    }

    { // Non-Empty Tests //
        { // Uni-Domain Test //
            Node mesh;
            blueprint::mesh::examples::braid("quads",10,10,1,mesh);

            const std::vector<Node *> domains = blueprint::mesh::domains(mesh);
            ASSERT_EQ(domains.size(), 1);
            ASSERT_EQ(domains.back(), &mesh);
        }

        { // Multi-Domain Test //
            Node mesh;
            blueprint::mesh::examples::grid("quads",10,10,1,2,2,1,mesh);

            std::set<Node *> ref_domains;
            for(const std::string &child_name : mesh.child_names())
            {
                ref_domains.insert(&mesh.child(child_name));
            }

            const std::vector<Node *> domains = blueprint::mesh::domains(mesh);
            std::set<Node *> acc_domains(domains.begin(), domains.end());
            ASSERT_EQ(domains.size(), ref_domains.size());
            ASSERT_EQ(acc_domains.size(), ref_domains.size());
            ASSERT_EQ(acc_domains, ref_domains);
        }
    }
}


//-----------------------------------------------------------------------------
TEST(conduit_blueprint_mesh_query, adjset_formats)
{
    { // Pairwise Tests //
        { // Empty Test //
            Node mesh, info;

            blueprint::mesh::examples::grid("quads",10,10,1,2,2,1,mesh);

            for(Node *domain : blueprint::mesh::domains(mesh))
            {
                Node &domain_adjset = (*domain)["adjsets"].child(0);
                domain_adjset["groups"].reset();
                domain_adjset["groups"].set(DataType::object());

                ASSERT_TRUE(blueprint::mesh::adjset::verify(domain_adjset, info));
                ASSERT_TRUE(blueprint::mesh::adjset::is_pairwise(domain_adjset));
            }
        }

        { // Positive Test //
            Node mesh, info;

            blueprint::mesh::examples::grid("quads",10,10,1,2,1,1,mesh);
            for(const Node *domain : blueprint::mesh::domains(mesh))
            {
                const Node &domain_adjset = (*domain)["adjsets"].child(0);
                ASSERT_TRUE(blueprint::mesh::adjset::verify(domain_adjset, info));
                ASSERT_TRUE(blueprint::mesh::adjset::is_pairwise(domain_adjset));
            }

            blueprint::mesh::examples::grid("quads",10,10,1,4,1,1,mesh);
            for(const Node *domain : blueprint::mesh::domains(mesh))
            {
                const Node &domain_adjset = (*domain)["adjsets"].child(0);
                ASSERT_TRUE(blueprint::mesh::adjset::verify(domain_adjset, info));
                ASSERT_TRUE(blueprint::mesh::adjset::is_pairwise(domain_adjset));
            }
        }

        { // Negative Test //
            Node mesh, info;

            blueprint::mesh::examples::grid("quads",10,10,1,2,2,1,mesh);
            for(const Node *domain : blueprint::mesh::domains(mesh))
            {
                const Node &domain_adjset = (*domain)["adjsets"].child(0);
                ASSERT_TRUE(blueprint::mesh::adjset::verify(domain_adjset, info));
                ASSERT_FALSE(blueprint::mesh::adjset::is_pairwise(domain_adjset));
            }

            blueprint::mesh::examples::grid("hexs",10,10,10,2,2,2,mesh);
            for(const Node *domain : blueprint::mesh::domains(mesh))
            {
                const Node &domain_adjset = (*domain)["adjsets"].child(0);
                ASSERT_TRUE(blueprint::mesh::adjset::verify(domain_adjset, info));
                ASSERT_FALSE(blueprint::mesh::adjset::is_pairwise(domain_adjset));
            }
        }
    }

    { // Max-Share Tests //
        { // Empty Test //
            Node mesh, info;

            blueprint::mesh::examples::grid("quads",10,10,1,2,2,1,mesh);

            for(Node *domain : blueprint::mesh::domains(mesh))
            {
                Node &domain_adjset = (*domain)["adjsets"].child(0);
                domain_adjset["groups"].reset();
                domain_adjset["groups"].set(DataType::object());

                ASSERT_TRUE(blueprint::mesh::adjset::verify(domain_adjset, info));
                ASSERT_TRUE(blueprint::mesh::adjset::is_maxshare(domain_adjset));
            }
        }

        { // Positive Test //
            Node mesh, info;

            blueprint::mesh::examples::grid("quads",10,10,1,2,1,1,mesh);
            for(const Node *domain : blueprint::mesh::domains(mesh))
            {
                const Node &domain_adjset = (*domain)["adjsets"].child(0);
                ASSERT_TRUE(blueprint::mesh::adjset::verify(domain_adjset, info));
                ASSERT_TRUE(blueprint::mesh::adjset::is_maxshare(domain_adjset));
            }

            blueprint::mesh::examples::grid("quads",10,10,1,2,2,1,mesh);
            for(const Node *domain : blueprint::mesh::domains(mesh))
            {
                const Node &domain_adjset = (*domain)["adjsets"].child(0);
                ASSERT_TRUE(blueprint::mesh::adjset::verify(domain_adjset, info));
                ASSERT_TRUE(blueprint::mesh::adjset::is_maxshare(domain_adjset));
            }
        }

        { // Negative Test //
            Node mesh, info;

            // TODO: All tests return adjsets that are natively 'max-share',
            // so testing the negative case will require either using a transform
            // or constructing a simple non-max-share mesh.
            // ASSERT_TRUE(true);
        }
    }
}
