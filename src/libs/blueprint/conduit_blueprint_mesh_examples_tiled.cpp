// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mesh_examples_tiled.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_BLUEPRINT_MESH_EXAMPLES_TILED_HPP
#define CONDUIT_BLUEPRINT_MESH_EXAMPLES_TILED_HPP

//-----------------------------------------------------------------------------
// conduit lib includes
//-----------------------------------------------------------------------------
#include "conduit_blueprint_mesh_examples_tiled.hpp"
#include "conduit.hpp"
#include "conduit_blueprint.hpp"
#include "conduit_blueprint_exports.h"
#include "conduit_blueprint_mesh_utils.hpp"
#include <cmath>

// Uncomment this to add some fields on the filed mesh prior to reordering.
// #define CONDUIT_TILER_DEBUG_FIELDS

// Uncomment this to try an experimental mode that uses the partitioner to
// do reordering.
// #define CONDUIT_USE_PARTITIONER_FOR_REORDER

// Uncomment this to use a simpler tiled pattern for debugging.
// #define CONDUIT_SIMPLE_TILED_PATTERN

//-----------------------------------------------------------------------------
// -- begin conduit::--
//-----------------------------------------------------------------------------
namespace conduit
{

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint --
//-----------------------------------------------------------------------------
namespace blueprint
{

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mesh --
//-----------------------------------------------------------------------------
namespace mesh
{

//-----------------------------------------------------------------------------
/// Methods that generate example meshes.
//-----------------------------------------------------------------------------
namespace examples
{

//-----------------------------------------------------------------------------
/// Detail
//-----------------------------------------------------------------------------
namespace detail
{

/**
 \brief Keep track of some tile information.
 */
class Tile
{
public:
    static const conduit::index_t INVALID_POINT;

    Tile() : ptids()
    {
    }

    /// Reset the tile. 
    void reset(size_t npts)
    {
        ptids = std::vector<conduit::index_t>(npts, -1);
    }

    /// Return the point ids.
          std::vector<conduit::index_t> &getPointIds() { return ptids; }
    const std::vector<conduit::index_t> &getPointIds() const { return ptids; }

    /// Get the specified point ids for this tile using the supplied indices.
    std::vector<conduit::index_t> getPointIds(const std::vector<conduit::index_t> &indices) const
    {
        std::vector<conduit::index_t> ids;
        ids.reserve(indices.size());
        for(const auto &idx : indices)
           ids.push_back(ptids[idx]);
        return ids;
    }

    // Set the point ids
    void setPointIds(const std::vector<conduit::index_t> &indices, const std::vector<conduit::index_t> &ids)
    {
        for(size_t i = 0; i < indices.size(); i++)
        {
            ptids[indices[i]] = ids[i];
        }
    }

private:
    std::vector<conduit::index_t> ptids;  //!< This tile's point ids.
};

const conduit::index_t Tile::INVALID_POINT = -1;

/**
 \brief Build a mesh from tiles. There is a default tile pattern, although it can
        be replaced using an options Node containing new tile information.
 */
class Tiler
{
public:
    static const int BoundaryLeft = 0;
    static const int BoundaryRight = 1;
    static const int BoundaryBottom = 2;
    static const int BoundaryTop = 3;
    static const int BoundaryBack = 4;
    static const int BoundaryFront = 5;

    static const conduit::index_t InvalidDomain = -1;

    Tiler();

    /// Generate the tiled mesh.
    void generate(conduit::index_t nx, conduit::index_t ny, conduit::index_t nz,
                  conduit::Node &res,
                  const conduit::Node &options);
protected:
    /// Fill a default tile pattern into the filter.
    void initialize();

    /// Fill in the tile pattern from a Node.
    void initialize(const conduit::Node &t);

    /// Return the topology (the first one)
    const conduit::Node &getTopology() const
    {
        const conduit::Node &t = m_options.fetch_existing("topologies");
        return t[0];
    }

    /// Return the coordset
    const conduit::Node &getCoordset() const
    {
        const conduit::Node &t = getTopology();
        std::string coordsetName(t.fetch_existing("coordset").as_string());
        return m_options.fetch_existing("coordsets/" + coordsetName);
    }

    /// Return point indices of points along left edge.
    const std::vector<conduit::index_t> &left() const { return m_left; }

    /// Return point indices of points along right edge.
    const std::vector<conduit::index_t> &right() const { return m_right; }

    /// Return point indices of points along bottom edge.
    const std::vector<conduit::index_t> &bottom() const { return m_bottom; }

    /// Return point indices of points along top edge.
    const std::vector<conduit::index_t> &top() const { return m_top; }

    /// Return tile width
    double width() const { return m_width; }

    /// Return tile height
    double height() const { return m_height; }

    /// Creates the points for the tile (if they need to be created).
    void addPoints(const double M[3][3],
                   std::vector<conduit::index_t> &ptids,
                   std::vector<double> &x,
                   std::vector<double> &y)
    {
        // Iterate through points in the template and add them if they have
        // not been created yet.
        const auto &xpts = getCoordset().fetch_existing("values/x").as_double_array();
        const auto &ypts = getCoordset().fetch_existing("values/y").as_double_array();
        for(conduit::index_t i = 0; i < xpts.number_of_elements(); i++)
        {
            if(ptids[i] == Tile::INVALID_POINT)
            {
                ptids[i] = static_cast<int>(x.size());

                // (x,y,1) * M
                double xc = xpts[i] * M[0][0] + ypts[i] * M[1][0] + M[2][0];
                double yc = xpts[i] * M[0][1] + ypts[i] * M[1][1] + M[2][1];
                double h  = xpts[i] * M[0][2] + ypts[i] * M[1][2] + M[2][2];
                xc /= h;
                yc /= h;
                x.push_back(xc);
                y.push_back(yc);
            }
        }
    }

    /// Iterate over faces
    template <typename Connectivity, typename Body>
    void iterateFaces(const Connectivity &conn,
                      conduit::index_t nelem,
                      conduit::index_t sides,
                      conduit::index_t *idlist,
                      const std::vector<conduit::index_t> &ptids,
                      conduit::index_t offset,
                      bool reverse,
                      int stype,
                      Body &&body) const
    {
        if(reverse)
        {
            auto s1 = sides - 1;
            for(conduit::index_t i = 0; i < nelem; i++)
            {
                auto start = i * sides;
                for(conduit::index_t s = 0; s < sides; s++)
                    idlist[s1 - s] = offset + ptids[conn[start + s]];
                body(idlist, sides, stype);
            }
        }
        else
        {
            for(conduit::index_t i = 0; i < nelem; i++)
            {
                auto start = i * sides;
                for(conduit::index_t s = 0; s < sides; s++)
                    idlist[s] = offset + ptids[conn[start + s]];
                body(idlist, sides, stype);
            }
        }
    }

    /// Iterate over the tile's elements and apply a lambda.
    template <typename Body>
    void iterateFaces(const std::vector<conduit::index_t> &ptids,
                      conduit::index_t offset,
                      bool reverse,
                      int stype,
                      Body &&body) const
    {
        const conduit::Node &topo = getTopology();
        std::string shape = topo.fetch_existing("elements/shape").as_string();
        conduit::index_t sides = 0;
        if(shape == "tri")
            sides = 3;
        else if(shape == "quad")
            sides = 4;

        // Handle triangles and quads.
        const conduit::Node &n_conn = topo.fetch_existing("elements/connectivity");
        if(sides == 3 || sides == 4)
        {
            conduit::index_t idlist[4];
            bool handled = false;
            if(n_conn.dtype().spanned_bytes() == n_conn.dtype().strided_bytes())
            {
                if(n_conn.dtype().is_index_t())
                {
                    iterateFaces(n_conn.as_index_t_ptr(),
                                 n_conn.dtype().number_of_elements() / sides,
                                 sides, idlist, ptids,
                                 offset, reverse, stype, body);
                    handled = true;
                }
                else if(n_conn.dtype().is_int32())
                {
                    iterateFaces(n_conn.as_int32_ptr(),
                                 n_conn.dtype().number_of_elements() / sides,
                                 sides, idlist, ptids,
                                 offset, reverse, stype, body);
                    handled = true;
                }
            }
            if(!handled)
            {
                iterateFaces(n_conn.as_index_t_accessor(),
                             n_conn.dtype().number_of_elements() / sides,
                             sides, idlist, ptids,
                             offset, reverse, stype, body);
            }
        }
        else if(shape == "polygonal")
        {
            // handle polygons
            const auto conn = n_conn.as_index_t_accessor();
            const auto sizes = topo.fetch_existing("elements/sizes").as_index_t_accessor();
            const conduit::index_t nelem = sizes.number_of_elements();
            conduit::index_t start = 0;
            std::vector<index_t> idlist(10);
            if(reverse)
            {
                for(conduit::index_t i = 0; i < nelem; i++)
                {
                    auto esides = sizes[i];
                    idlist.reserve(esides);
                    for(conduit::index_t s = 0; s < esides; s++)
                        idlist[s] = offset + ptids[conn[start + esides - s]];
                    body(&idlist[0], esides, stype);
                    start += esides;
                }
            }
            else
            {
                for(conduit::index_t i = 0; i < nelem; i++)
                {
                    auto esides = sizes[i];
                    idlist.reserve(esides);
                    for(conduit::index_t s = 0; s < esides; s++)
                        idlist[s] = offset + ptids[conn[start + s]];
                    body(&idlist[0], esides, stype);
                    start += esides;
                }
            }
        }
    }

    /// Emit the hex cells using this tile's point ids.
    void addVolumeElements(const std::vector<conduit::index_t> &ptids,
                 conduit::index_t plane1Offset,
                 conduit::index_t plane2Offset,
                 std::vector<conduit::index_t> &conn,
                 std::vector<conduit::index_t> &sizes) const
    {
        const conduit::Node &topo = getTopology();
        std::string shape = topo.fetch_existing("elements/shape").as_string();
        conduit::index_t sides = 0;
        if(shape == "tri")
            sides = 3;
        else if(shape == "quad")
            sides = 4;

        if(sides == 3 || sides == 4)
        {
            const conduit::Node &n_conn = topo.fetch_existing("elements/connectivity");
            const auto tileconn = n_conn.as_index_t_accessor();
            const conduit::index_t nelem = tileconn.number_of_elements() / sides;
            for(conduit::index_t i = 0; i < nelem; i++)
            {
                conduit::index_t start = i * sides;
                for(conduit::index_t s = 0; s < sides; s++)
                    conn.push_back(plane1Offset + ptids[tileconn[start + s]]);

                for(conduit::index_t s = 0; s < sides; s++)
                    conn.push_back(plane2Offset + ptids[tileconn[start + s]]);

                sizes.push_back(2 * sides);
            }
        }
        else
        {
            CONDUIT_ERROR("Tiling polygonal shapes into 3D polyhedra is not yet supported.");
        }
    }

    /// Turn a node into an int vector.
    std::vector<conduit::index_t> toIndexVector(const conduit::Node &n) const
    {
        auto acc = n.as_index_t_accessor();
        std::vector<index_t> vec;
        vec.reserve(acc.number_of_elements());
        for(conduit::index_t i = 0; i < acc.number_of_elements(); i++)
            vec.push_back(acc[i]);
        return vec;
    }

    /// Determine which boundaries are needed.
    void boundaryFlags(const conduit::Node &options, bool flags[6]) const;

    // Iterate over 2D boundaries
    template <typename Body>
    void iterateBoundary2D(const std::vector<Tile> &tiles,
                           conduit::index_t nx,
                           conduit::index_t ny,
                           const bool flags[6],
                           Body &&body) const;

    /// Iterate over 3D boundaries.
    template <typename Body>
    void iterateBoundary3D(const std::vector<Tile> &tiles,
                           conduit::index_t nx,
                           conduit::index_t ny,
                           conduit::index_t nz,
                           conduit::index_t nPtsPerPlane,
                           const bool flags[6],
                           Body &&body) const;

    /// Compute domain id, if it exists, or return InvalidDomain.
    conduit::index_t domainIndex(conduit::index_t d0, conduit::index_t d1, conduit::index_t d2,
                                  const std::vector<conduit::index_t> &domain,
                                  const std::vector<conduit::index_t> &domains) const;

    /// Compute domain id, if it exists, or return InvalidDomain.
    conduit::index_t domainIndex(const conduit::index_t delta[3],
                                  const std::vector<conduit::index_t> &domain,
                                  const std::vector<conduit::index_t> &domains) const;

    /// Add ptids to a values node for an adjset, possibly transforming the point ids.
    void addAdjsetValues(const std::vector<conduit::index_t> &ptids,
                         conduit::index_t planeOffset,
                         bool reorder,
                         const std::vector<conduit::index_t> &old2NewPoint,
                         conduit::Node &values) const;

    /// Add adjacency set
    void addAdjset(const std::vector<Tile> &tiles,
                   conduit::index_t nx,
                   conduit::index_t ny,
                   conduit::index_t nz,
                   conduit::index_t ptsPerPlane,
                   bool reorder,
                   const std::vector<conduit::index_t> &old2NewPoint,
                   const conduit::Node &options,
                   conduit::Node &out) const;
private:
    conduit::Node m_options;
    double m_width, m_height;
    std::vector<conduit::index_t> m_left, m_right, m_bottom, m_top;
    std::string meshName, boundaryMeshName;
};

//---------------------------------------------------------------------------
Tiler::Tiler() : m_options(), m_width(0.), m_height(0.),
                 m_left(), m_right(), m_bottom(), m_top(),
                 meshName("mesh"), boundaryMeshName("boundary")
{
    initialize();
}

//---------------------------------------------------------------------------
void
Tiler::initialize()
{
#ifdef CONDUIT_SIMPLE_TILED_PATTERN
    // Simpler case for debugging.
    const double x[] = {0., 1., 0., 1.};
    const double y[] = {0., 0., 1., 1.};
    const conduit::index_t conn[] = {0,1,3,2};

    const conduit::index_t left[] = {0,2};
    const conduit::index_t right[] = {1,3};
    const conduit::index_t bottom[] = {0,1};
    const conduit::index_t top[] = {2,3};
#else
    // Default pattern
    const double x[] = {
        0., 3., 10., 17., 20.,
        0., 3., 17., 20.,
        5., 15.,
        7., 10., 13.,
        0., 7., 10., 13., 20.,
        7., 10., 13.,
        5., 15.,
        0., 3., 17., 20.,
        0., 3., 10., 17., 20.,
    };

    const double y[] = {
        0., 0., 0., 0., 0.,
        3., 3., 3., 3.,
        5., 5.,
        7., 7., 7.,
        10., 10., 10., 10., 10.,
        13., 13., 13.,
        15., 15.,
        17., 17., 17., 17.,
        20., 20., 20., 20., 20.,
    };

    const conduit::index_t conn[] = {
        // lower-left quadrant
        0,1,6,5,
        1,2,9,6,
        2,12,11,9,
        5,6,9,14,
        9,11,15,14,
        11,12,16,15,
        // lower-right quadrant
        2,3,7,10,
        3,4,8,7,
        7,8,18,10,
        2,10,13,12,
        12,13,17,16,
        10,18,17,13,
        // upper-left quadrant
        14,22,25,24,
        14,15,19,22,
        15,16,20,19,
        24,25,29,28,
        22,30,29,25,
        19,20,30,22,
        // upper-right quadrant
        16,17,21,20,
        17,18,23,21,
        18,27,26,23,
        20,21,23,30,
        23,26,31,30,
        26,27,32,31
    };

    const conduit::index_t left[] = {0,5,14,24,28};
    const conduit::index_t right[] = {4,8,18,27,32};
    const conduit::index_t bottom[] = {0,1,2,3,4};
    const conduit::index_t top[] = {28,29,30,31,32};
#endif

    // Define the tile as a topology.
    conduit::Node opts;
    opts["coordsets/coords/type"] = "explicit";
    opts["coordsets/coords/values/x"].set(x, sizeof(x) / sizeof(double));
    opts["coordsets/coords/values/y"].set(y, sizeof(y) / sizeof(double));
    opts["topologies/tile/type"] = "unstructured";
    opts["topologies/tile/coordset"] = "coords";
    opts["topologies/tile/elements/shape"] = "quad";
    constexpr auto nelem = (sizeof(conn) / sizeof(conduit::index_t)) / 4;
    opts["topologies/tile/elements/connectivity"].set(conn, nelem * 4);
    std::vector<conduit::index_t> size(nelem, 4);
    opts["topologies/tile/elements/sizes"].set(size.data(), size.size());

    // Define tile boundary indices.
    opts["left"].set(left, sizeof(left) / sizeof(conduit::index_t));
    opts["right"].set(right, sizeof(right) / sizeof(conduit::index_t));
    opts["bottom"].set(bottom, sizeof(bottom) / sizeof(conduit::index_t));
    opts["top"].set(top, sizeof(top) / sizeof(conduit::index_t));

    initialize(opts);
}

//---------------------------------------------------------------------------
void
Tiler::initialize(const conduit::Node &t)
{
    std::vector<std::string> required{"coordsets", "topologies", "left", "right", "bottom", "top"};
    for(const auto &name : required)
    {
        if(!t.has_child(name))
        {
            CONDUIT_ERROR("Node does not contain key: " << name);
        }
    }

    // Get the tile boundaries and convert them.
    m_left = toIndexVector(t.fetch_existing("left"));
    m_right = toIndexVector(t.fetch_existing("right"));
    m_bottom = toIndexVector(t.fetch_existing("bottom"));
    m_top = toIndexVector(t.fetch_existing("top"));
    if(m_left.size() != m_right.size())
    {
        CONDUIT_ERROR("left/right vectors have different lengths.");
    }
    if(m_bottom.size() != m_top.size())
    {
        CONDUIT_ERROR("bottom/top vectors have different lengths.");
    }

    // Save the tile definition options.
    m_options.set(t);

    // Make sure the coordset is 2D, explicit.
    if(conduit::blueprint::mesh::coordset::dims(getCoordset()) != 2)
    {
        CONDUIT_ERROR("The tile coordset must be 2D.");
    }
    if(getCoordset()["type"].as_string() != "explicit")
    {
        CONDUIT_ERROR("The tile coordset must be explicit.");
    }
    // Make sure the topology is 2D, unstructured
    if(conduit::blueprint::mesh::topology::dims(getTopology()) != 2)
    {
        CONDUIT_ERROR("The tile topology must be 2D.");
    }
    if(getTopology()["type"].as_string() != "unstructured")
    {
        CONDUIT_ERROR("The tile topology must be 2D.");
    }

    // Compute the tile extents.
    if(t.has_path("translate/x"))
        m_width = t["translate/x"].to_double();
    else
    {
        const auto &xc = getCoordset().fetch_existing("values/x").as_double_array();
        m_width = xc.max() - xc.min();
    }

    if(t.has_path("translate/y"))
         m_height = t["translate/y"].to_double();
    else
    {
        const auto &yc = getCoordset().fetch_existing("values/y").as_double_array();
        m_height = yc.max() - yc.min();
    }
}

//---------------------------------------------------------------------------
/**
 \brief Generate coordinate and connectivity arrays using a tiled mesh pattern,
        given by the Tile class.

 \param nx The number of tiles in the X dimension.
 \param ny The number of tiles in the Y dimension.
 \param nz The number of tiles in the Z dimension.
 \param[out] res The output node.
 \param options A node that may contain additional control options.
 */
void
Tiler::generate(conduit::index_t nx, conduit::index_t ny, conduit::index_t nz,
    conduit::Node &res,
    const conduit::Node &options)
{
    std::vector<double> x, y, z;
    std::vector<conduit::index_t> conn, sizes, bconn, bsizes;
    std::vector<int> btype;

    // Process any options.
    if(options.has_path("tile"))
        initialize(options.fetch_existing("tile"));

    bool reorder = true;
    if(options.has_path("reorder"))
        reorder = options.fetch_existing("reorder").to_int() > 0;

    if(options.has_path("meshname"))
        meshName = options.fetch_existing("meshname").as_string();
    if(options.has_path("boundarymeshname"))
        boundaryMeshName = options.fetch_existing("boundarymeshname").as_string();

    conduit::DataType indexDT(conduit::DataType::index_t());
    if(options.has_child("datatype"))
    {
        auto s = options.fetch_existing("datatype").as_string();
        if((s == "int") || (s == "int32") || (s == "integer"))
        {
            indexDT = conduit::DataType::int32();
        }
    }

    // Make a transformation matrix for the tile points.
    double origin[] = {0., 0., 0.};
    double tx = width(), ty = height();
    double z1 = std::max(width(), height()) * nz;
    double M[3][3] = {{1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}};
    if(options.has_path("extents"))
    {
        // Extents for the domain were given. Fit the domain into it.
        auto extents = options.fetch_existing("extents").as_double_accessor();
        tx = (extents[1] - extents[0]) / nx;
        ty = (extents[3] - extents[2]) / ny;
        origin[0] = extents[0];
        origin[1] = extents[2];
        origin[2] = extents[4];
        z1 = extents[5];
    }
    else
    {
        // There are no extents so figure out some based on the domains, if present.
        if(options.has_path("domain") && options.has_path("domains"))
        {
            auto domain = options.fetch_existing("domain").as_int_accessor();
            auto domains = options.fetch_existing("domains").as_int_accessor();
            if(domain.number_of_elements() == 3 &&
               domain.number_of_elements() == domains.number_of_elements())
            {
                origin[0] = domain[0] * nx * width();
                origin[1] = domain[1] * ny * height();
                origin[2] = domain[2] * z1;
                z1 = origin[2] + z1;
            }
        }
    }
    // Scaling
    M[0][0] = tx / width();
    M[1][1] = ty / height();
    // Translation
    M[2][0] = origin[0];
    M[2][1] = origin[1];

    // Number of tile points.
    const auto nTilePts = getCoordset().fetch_existing("values/x").dtype().number_of_elements();

    // Make a pass where we make nx*ny tiles so we can generate their points.
    std::vector<Tile> tiles(nx * ny);
    for(conduit::index_t j = 0; j < ny; j++)
    {
        M[2][0] = origin[0];
        for(conduit::index_t i = 0; i < nx; i++)
        {
            Tile &current = tiles[(j*nx + i)];

            // The first time we've used the tile, set its size.
            current.reset(nTilePts);

            // Copy some previous points over so they can be shared.
            if(i > 0)
            {
               Tile &prevX = tiles[(j*nx + i - 1)];
               current.setPointIds(left(), prevX.getPointIds(right()));
            }
            if(j > 0)
            {
               Tile &prevY = tiles[((j-1)*nx + i)];
               current.setPointIds(bottom(), prevY.getPointIds(top()));
            }

            addPoints(M, current.getPointIds(), x, y);
            M[2][0] += tx;
        }
        M[2][1] += ty;
    }

    conduit::index_t ptsPerPlane = 0;
    std::string shape2, shape3;
    shape2 = getTopology().fetch_existing("elements/shape").as_string();
    if(nz < 1)
    {
        // Iterate over the tiles and add their quads.
        // TODO: reserve size for conn, sizes
        for(conduit::index_t j = 0; j < ny; j++)
        {
            for(conduit::index_t i = 0; i < nx; i++)
            {
                Tile &current = tiles[(j*nx + i)];
                iterateFaces(current.getPointIds(), 0, false, BoundaryBack,
                    [&](const conduit::index_t *ids, conduit::index_t npts, int)
                    {
                        for(conduit::index_t pi = 0; pi < npts; pi++)
                            conn.push_back(ids[pi]);
                        sizes.push_back(npts);
                    });
            }
        }
        // NOTE: z coords in output will be empty.
    }
    else
    {
        ptsPerPlane = static_cast<conduit::index_t>(x.size());

        shape3 = (shape2 == "tri") ? "wedge" : "hex";

        // We have x,y points now. We need to replicate them to make multiple planes.
        // We make z coordinates too.
        conduit::index_t nplanes = nz + 1;
        x.reserve(ptsPerPlane * nplanes);
        y.reserve(ptsPerPlane * nplanes);
        z.reserve(ptsPerPlane * nplanes);
        for(conduit::index_t i = 0; i < ptsPerPlane; i++)
            z.push_back(origin[2]);
        for(conduit::index_t p = 1; p < nplanes; p++)
        {
            double t = static_cast<double>(p) / static_cast<double>(nplanes - 1);
            double zvalue = (1. - t) * origin[2] + t * z1;
            for(conduit::index_t i = 0; i < ptsPerPlane; i++)
            {
                x.push_back(x[i]);
                y.push_back(y[i]);
                z.push_back(zvalue);
            }
        }

        // Iterate over the tiles and add their hexs.
        // TODO: reserve size for conn, sizes
        for(conduit::index_t k = 0; k < nz; k++)
        {
            conduit::index_t offset1 = k * ptsPerPlane;
            conduit::index_t offset2 = offset1 + ptsPerPlane;

            for(conduit::index_t j = 0; j < ny; j++)
            {
                for(conduit::index_t i = 0; i < nx; i++)
                {
                    Tile &current = tiles[(j*nx + i)];
                    addVolumeElements(current.getPointIds(), offset1, offset2, conn, sizes);
                }
            }
        }
    }

    // Make the Blueprint mesh.
    res["coordsets/coords/type"] = "explicit";
    res["coordsets/coords/values/x"].set(x);
    res["coordsets/coords/values/y"].set(y);
    if(!z.empty())
        res["coordsets/coords/values/z"].set(z);

    conduit::Node &topo = res["topologies/" + meshName];
    topo["type"] = "unstructured";
    topo["coordset"] = "coords";
    topo["elements/shape"] = z.empty() ? shape2 : shape3;
    conduit::Node tmp;
    tmp.set_external(conn.data(), conn.size());
    tmp.to_data_type(indexDT.id(), topo["elements/connectivity"]);
    tmp.set_external(sizes.data(), sizes.size());
    tmp.to_data_type(indexDT.id(), topo["elements/sizes"]);

#ifdef CONDUIT_TILER_DEBUG_FIELDS
    // Add fields to test the reordering.
    std::vector<conduit::index_t> nodeids, elemids;
    auto npts = static_cast<conduit::index_t>(x.size());
    nodeids.reserve(npts);
    for(conduit::index_t i = 0; i < npts; i++)
        nodeids.push_back(i);
    res["fields/nodeids/topology"] = meshName;
    res["fields/nodeids/association"] = "vertex";
    res["fields/nodeids/values"].set(nodeids);

    auto nelem = static_cast<conduit::index_t>(sizes.size());
    elemids.reserve(nelem);
    for(conduit::index_t i = 0; i < nelem; i++)
        elemids.push_back(i);
    res["fields/elemids/topology"] = meshName;
    res["fields/elemids/association"] = "element";
    res["fields/elemids/values"].set(elemids);

    std::vector<double> dist;
    dist.reserve(npts);
    if(nz < 1)
    {
        for(conduit::index_t i = 0; i < npts; i++)
            dist.push_back(sqrt(x[i]*x[i] + y[i]*y[i]));
    }
    else
    {
        for(conduit::index_t i = 0; i < npts; i++)
            dist.push_back(sqrt(x[i]*x[i] + y[i]*y[i] + z[i]*z[i]));
    }
    res["fields/dist/topology"] = meshName;
    res["fields/dist/association"] = "vertex";
    res["fields/dist/values"].set(dist);
#endif

    // Reorder the elements unless it was turned off.
    std::vector<conduit::index_t> old2NewPoint;
    if(reorder)
    {
        // We need offsets.
        conduit::blueprint::mesh::utils::topology::unstructured::generate_offsets(topo, topo["elements/offsets"]);

        // Create a new order for the mesh elements.
        const auto elemOrder = conduit::blueprint::mesh::utils::topology::spatial_ordering(topo);

#ifdef CONDUIT_USE_PARTITIONER_FOR_REORDER
        // NOTE: This was an idea I had after I made reorder. Reordering is like
        //       making an explicit selection for the partitioner. Can we just use
        //       the partitioner? Kind of, it turns out.
        //
        //       1. Elements are reordered like we want
        //       2. Nodes are not reordered in their order of use by elements.
        //       3. Passing the same node as input/output does bad things.

        // Make an explicit selection for partition to do the reordering.
        conduit::Node options;
        conduit::Node &sel = options["selections"].append();
        sel["type"] = "explicit";
        sel["topology"] = meshName;
        sel["elements"].set_external(const_cast<conduit::index_t *>(elemOrder.data()), elemOrder.size());
        conduit::Node output;
        conduit::blueprint::mesh::partition(res, options, output);

        // Extract the vertex mapping.
        auto ids = output.fetch_existing("fields/original_vertex_ids/values/ids").as_index_t_accessor();
        for(conduit::index_t i = 0; i < ids.number_of_elements(); i++)
        {
            old2NewPoint.push_back(ids[i]);
        }
        res.reset();
        res.move(output);
#else
        conduit::blueprint::mesh::utils::topology::unstructured::reorder(
            topo, res["coordsets/coords"], res["fields"],
            elemOrder,
            topo, res["coordsets/coords"], res["fields"],
            old2NewPoint);
#endif
    }

    // Boundaries
    std::string bshape;
    bool flags[6];
    boundaryFlags(options, flags);
    
    if(nz < 1)
    {
        // 2D
        bshape = "line";
        if(reorder)
        {
            iterateBoundary2D(tiles, nx, ny, flags,
                [&](const conduit::index_t *ids, conduit::index_t npts, int bnd)
                {
                    for(conduit::index_t i = 0; i < npts; i++)
                        bconn.push_back(old2NewPoint[ids[i]]); // Renumber
                    bsizes.push_back(npts);
                    btype.push_back(bnd + 1); // Make 1-origin
                });
        }
        else
        {
            iterateBoundary2D(tiles, nx, ny, flags,
                [&](const conduit::index_t *ids, conduit::index_t npts, int bnd)
                {
                    for(conduit::index_t i = 0; i < npts; i++)
                        bconn.push_back(ids[i]);
                    bsizes.push_back(npts);
                    btype.push_back(bnd + 1); // Make 1-origin
                });
        }
    }
    else
    {
        // 3D
        bshape = "quad";
        bool anyNonQuads = false;
        if(reorder)
        {
            iterateBoundary3D(tiles, nx, ny, nz, ptsPerPlane, flags,
                [&](const conduit::index_t *ids, conduit::index_t npts, int bnd)
                {
                    for(conduit::index_t i = 0; i < npts; i++)
                        bconn.push_back(old2NewPoint[ids[i]]); // Renumber
                    bsizes.push_back(npts);
                    btype.push_back(bnd + 1); // Make 1-origin
                    anyNonQuads |= (npts != 4);
                });
        }
        else
        {
            iterateBoundary3D(tiles, nx, ny, nz, ptsPerPlane, flags,
                [&](const conduit::index_t *ids, conduit::index_t npts, int bnd)
                {
                    for(conduit::index_t i = 0; i < npts; i++)
                        bconn.push_back(ids[i]);
                    bsizes.push_back(npts);
                    btype.push_back(bnd + 1); // Make 1-origin
                    anyNonQuads |= (npts != 4);
                });
        }
        if(anyNonQuads)
            bshape = "polygonal";
    }
    if(!bconn.empty())
    {
        conduit::Node &btopo = res["topologies/" + boundaryMeshName];
        btopo["type"] = "unstructured";
        btopo["coordset"] = "coords";
        btopo["elements/shape"] = bshape;

        tmp.set_external(bconn.data(), bconn.size());
        tmp.to_data_type(indexDT.id(), btopo["elements/connectivity"]);

        tmp.set_external(bsizes.data(), bsizes.size());
        tmp.to_data_type(indexDT.id(), btopo["elements/sizes"]);

        if(bshape == "polygonal")
            conduit::blueprint::mesh::utils::topology::unstructured::generate_offsets(btopo, btopo["elements/offsets"]);

        res["fields/boundary_attribute/topology"] = boundaryMeshName;
        res["fields/boundary_attribute/association"] = "element";
        res["fields/boundary_attribute/values"].set(btype);
    }

    // Build an adjacency set.
    addAdjset(tiles, nx, ny, nz, ptsPerPlane, reorder, old2NewPoint, options, res);

#if 0
    // Print for debugging.
    conduit::Node opts;
    opts["num_children_threshold"] = 100000;
    opts["num_elements_threshold"] = 500;
    std::cout << res.to_summary_string(opts) << std::endl;
#endif
}

//---------------------------------------------------------------------------
void
Tiler::boundaryFlags(const conduit::Node &options, bool flags[6]) const
{
    bool handled = false;
    if(options.has_path("domain") && options.has_path("domains"))
    {
        auto domain = options.fetch_existing("domain").as_int_accessor();
        auto domains = options.fetch_existing("domains").as_int_accessor();
        if(domain.number_of_elements() == 3 &&
           domain.number_of_elements() == domains.number_of_elements())
        {
            int ndoms = domains[0] * domains[1] * domains[2];
            if(ndoms > 1)
            {
                flags[BoundaryLeft]   = (domain[0] == 0);
                flags[BoundaryRight]  = (domain[0] == domains[0]-1);
                flags[BoundaryBottom] = (domain[1] == 0);
                flags[BoundaryTop]    = (domain[1] == domains[1]-1);
                flags[BoundaryBack]   = (domain[2] == 0);
                flags[BoundaryFront]  = (domain[2] == domains[2]-1);

                handled = true;
            }
        }
    }
    if(!handled)
    {
        for(int i = 0; i < 6; i++)
            flags[i] = true;
    }
}

//---------------------------------------------------------------------------
template <typename Body>
void
Tiler::iterateBoundary2D(const std::vector<Tile> &tiles,
    conduit::index_t nx,
    conduit::index_t ny,
    const bool flags[6],
    Body &&body) const
{
    conduit::index_t idlist[2];

    if(flags[BoundaryLeft])
    {
        for(conduit::index_t i = 0, j = 0; j < ny; j++)
        {
            const Tile &current = tiles[(j*nx + i)];
            const auto ids = current.getPointIds(left());
            for(size_t bi = 0; bi < ids.size() - 1; bi++)
            {
                idlist[0] = ids[bi];
                idlist[1] = ids[bi + 1];
                body(idlist, 2, BoundaryLeft);
            }
        }
    }
    if(flags[BoundaryRight])
    {
        for(conduit::index_t i = nx - 1, j = 0; j < ny; j++)
        {
            const Tile &current = tiles[(j*nx + i)];
            const auto ids = current.getPointIds(right());
            for(size_t bi = 0; bi < ids.size() - 1; bi++)
            {
                idlist[0] = ids[bi];
                idlist[1] = ids[bi + 1];
                body(idlist, 2, BoundaryRight);
            }
        }
    }
    if(flags[BoundaryBottom])
    {
        for(conduit::index_t i = 0, j = 0; i < nx; i++)
        {
            const Tile &current = tiles[(j*nx + i)];
            const auto ids = current.getPointIds(bottom());
            for(size_t bi = 0; bi < ids.size() - 1; bi++)
            {
                idlist[0] = ids[bi];
                idlist[1] = ids[bi + 1];
                body(idlist, 2, BoundaryBottom);
            }
        }
    }
    if(flags[BoundaryTop])
    {
        for(conduit::index_t i = 0, j = ny - 1; i < nx; i++)
        {
            const Tile &current = tiles[(j*nx + i)];
            const auto ids = current.getPointIds(top());
            for(size_t bi = 0; bi < ids.size() - 1; bi++)
            {
                idlist[0] = ids[bi];
                idlist[1] = ids[bi + 1];
                body(idlist, 2, BoundaryTop);
            }
        }
    }
}

//---------------------------------------------------------------------------
template <typename Body>
void
Tiler::iterateBoundary3D(const std::vector<Tile> &tiles,
    conduit::index_t nx,
    conduit::index_t ny,
    conduit::index_t nz,
    conduit::index_t nPtsPerPlane,
    const bool flags[6],
    Body &&body) const
{
    conduit::index_t idlist[4];

    if(flags[BoundaryLeft])
    {
        for(conduit::index_t k = 0; k < nz; k++)
        {
            conduit::index_t offset1 = k * nPtsPerPlane;
            conduit::index_t offset2 = (k + 1) * nPtsPerPlane;
            for(conduit::index_t i = 0, j = 0; j < ny; j++)
            {
                const Tile &current = tiles[(j*nx + i)];
                const auto ids = current.getPointIds(left());
                for(size_t bi = 0; bi < ids.size() - 1; bi++)
                {
                    idlist[0] = offset1 + ids[bi];
                    idlist[1] = offset2 + ids[bi];
                    idlist[2] = offset2 + ids[bi + 1];
                    idlist[3] = offset1 + ids[bi + 1];
                    body(idlist, 4, BoundaryLeft);
                }
            }
        }
    }
    if(flags[BoundaryRight])
    {
        for(conduit::index_t k = 0; k < nz; k++)
        {
            conduit::index_t offset1 = k * nPtsPerPlane;
            conduit::index_t offset2 = (k + 1) * nPtsPerPlane;
            for(conduit::index_t i = nx - 1, j = 0; j < ny; j++)
            {
                const Tile &current = tiles[(j*nx + i)];
                const auto ids = current.getPointIds(right());
                for(size_t bi = 0; bi < ids.size() - 1; bi++)
                {
                    idlist[0] = offset1 + ids[bi];
                    idlist[1] = offset1 + ids[bi + 1];
                    idlist[2] = offset2 + ids[bi + 1];
                    idlist[3] = offset2 + ids[bi];
                    body(idlist, 4, BoundaryRight);
                }
            }
        }
    }
    if(flags[BoundaryBottom])
    {
        for(conduit::index_t k = 0; k < nz; k++)
        {
            conduit::index_t offset1 = k * nPtsPerPlane;
            conduit::index_t offset2 = (k + 1) * nPtsPerPlane;
            for(conduit::index_t i = 0, j = 0; i < nx; i++)
            {
                const Tile &current = tiles[(j*nx + i)];
                const auto ids = current.getPointIds(bottom());
                for(size_t bi = 0; bi < ids.size() - 1; bi++)
                {
                    idlist[0] = offset1 + ids[bi];
                    idlist[1] = offset1 + ids[bi + 1];
                    idlist[2] = offset2 + ids[bi + 1];
                    idlist[3] = offset2 + ids[bi];
                    body(idlist, 4, BoundaryBottom);
                }
            }
        }
    }
    if(flags[BoundaryTop])
    {
        for(conduit::index_t k = 0; k < nz; k++)
        {
            conduit::index_t offset1 = k * nPtsPerPlane;
            conduit::index_t offset2 = (k + 1) * nPtsPerPlane;
            for(conduit::index_t i = 0, j = ny - 1; i < nx; i++)
            {
                const Tile &current = tiles[(j*nx + i)];
                const auto ids = current.getPointIds(top());
                for(size_t bi = 0; bi < ids.size() - 1; bi++)
                {
                    idlist[0] = offset1 + ids[bi];
                    idlist[1] = offset2 + ids[bi];
                    idlist[2] = offset2 + ids[bi + 1];
                    idlist[3] = offset1 + ids[bi + 1];
                    body(idlist, 4, BoundaryTop);
                }
            }
        }
    }
    if(flags[BoundaryBack])
    {
        for(conduit::index_t j = 0; j < ny; j++)
        for(conduit::index_t i = 0; i < nx; i++)
        {
           const Tile &current = tiles[(j*nx + i)];
           iterateFaces(current.getPointIds(), 0, true, BoundaryBack, body);
        }
    }
    if(flags[BoundaryFront])
    {
        for(conduit::index_t j = 0; j < ny; j++)
        for(conduit::index_t i = 0; i < nx; i++)
        {
           const Tile &current = tiles[(j*nx + i)];
           iterateFaces(current.getPointIds(), nz * nPtsPerPlane, false, BoundaryFront, body);
        }
    }
}

//---------------------------------------------------------------------------
conduit::index_t
Tiler::domainIndex(conduit::index_t d0, conduit::index_t d1, conduit::index_t d2,
                    const std::vector<conduit::index_t> &domain,
                    const std::vector<conduit::index_t> &domains) const
{
    const conduit::index_t dom[3] = {domain[0] + d0, domain[1] + d1, domain[2] + d2};
    // If the domain exists, make its domain id.
    conduit::index_t domainId = InvalidDomain;
    if((dom[0] >= 0 && dom[0] < domains[0]) &&
       (dom[1] >= 0 && dom[1] < domains[1]) &&
       (dom[2] >= 0 && dom[2] < domains[2]))
    {
        domainId = dom[2] * (domains[0] * domains[1]) + dom[1] * domains[0] + dom[0];
    }
    return domainId;
}

//---------------------------------------------------------------------------
conduit::index_t
Tiler::domainIndex(const conduit::index_t delta[3],
                    const std::vector<conduit::index_t> &domain,
                    const std::vector<conduit::index_t> &domains) const
{
    return domainIndex(delta[0], delta[1], delta[2], domain, domains);
}

//---------------------------------------------------------------------------
void
Tiler::addAdjsetValues(const std::vector<conduit::index_t> &ptids,
                       conduit::index_t planeOffset,
                       bool reorder,
                       const std::vector<conduit::index_t> &old2NewPoint,
                       conduit::Node &values) const
{
    if(reorder)
    {
        values.set(conduit::DataType::index_t(ptids.size()));
        conduit::index_t *dest = values.as_index_t_ptr();
        if(planeOffset > 0)
        {
            for(size_t i = 0; i < ptids.size(); i++)
                *dest++ = old2NewPoint[ptids[i] + planeOffset];
        }
        else
        {
            for(size_t i = 0; i < ptids.size(); i++)
                *dest++ = old2NewPoint[ptids[i]];
        }
    }
    else if(planeOffset > 0)
    {
        values.set(conduit::DataType::index_t(ptids.size()));
        conduit::index_t *dest = values.as_index_t_ptr();
        for(size_t i = 0; i < ptids.size(); i++)
            *dest++ = ptids[i] + planeOffset;
    }
    else
    {
        values.set(ptids);
    }
}

//---------------------------------------------------------------------------
void
Tiler::addAdjset(const std::vector<Tile> &tiles,
                 conduit::index_t nx,
                 conduit::index_t ny,
                 conduit::index_t nz,
                 conduit::index_t ptsPerPlane,
                 bool reorder,
                 const std::vector<conduit::index_t> &old2NewPoint,
                 const conduit::Node &options,
                 conduit::Node &out) const
{
    // Make the adjset name for 2 domains.
    auto adjset_name = [](conduit::index_t d0, conduit::index_t d1) {
        if(d0 > d1)
            std::swap(d0, d1);
        std::stringstream ss;
        ss << "domain_" << d0 << "_" << d1;
        return ss.str();
    };

    // Build up unique points, given a list of points. Use unique and ptvec by capture.
    std::set<conduit::index_t> unique;
    std::vector<conduit::index_t> ptvec;
    auto addPoints = [&](const conduit::index_t *ids, conduit::index_t npts, int /*bnd*/)
    {
        for(conduit::index_t i = 0; i < npts; i++)
        {
            const auto id = ids[i];
            if(unique.find(id) == unique.end())
            {
                unique.insert(id);
                ptvec.push_back(id);
            }
        }
    };

    // We need to know where this domain is in the domains to make the adjset.
    if(options.has_child("domain") && options.has_child("domains"))
    {
        auto domain = toIndexVector(options.fetch_existing("domain"));
        auto domains = toIndexVector(options.fetch_existing("domains"));
        if(domain.size() == 3 && domain.size() == domains.size())
        {
            if(domains[0] * domains[1] * domains[2] > 1)
            {
                auto thisDom = domainIndex(0, 0, 0, domain, domains);

                // Make a state node.
                out["state/domain_id"] = thisDom;

                // Make the top level adjset nodes.
                conduit::Node &adjset = out["adjsets/" + meshName + "_adjset"];
                adjset["association"] = "vertex";
                adjset["topology"] = meshName;
                conduit::Node &groups = adjset["groups"];

                // Trace around the tile to get the points for the edges of the tile
                // in original node order.
                //-----------------------------------------------------------
                std::vector<conduit::index_t> left, right, bottom, top;
                const bool left_flags[] = {true, false, false, false};
                iterateBoundary2D(tiles, nx, ny, left_flags, addPoints);
                left.swap(ptvec); // Steal the points.
                unique.clear();
                const bool right_flags[] = {false, true, false, false};
                iterateBoundary2D(tiles, nx, ny, right_flags, addPoints);
                right.swap(ptvec); // Steal the points.
                unique.clear();
                const bool bottom_flags[] = {false, false, true, false};
                iterateBoundary2D(tiles, nx, ny, bottom_flags, addPoints);
                bottom.swap(ptvec); // Steal the points.
                unique.clear();
                const bool top_flags[] = {false, false, false, true};
                iterateBoundary2D(tiles, nx, ny, top_flags, addPoints);
                top.swap(ptvec); // Steal the points.
                unique.clear();

                // Make corner neighbors.
                //-----------------------------------------------------------
                conduit::index_t frontPlane = ptsPerPlane * nz;
                conduit::index_t corner[8];
                corner[0] = left[0];                 // back, lower left
                corner[1] = right[0];                // back, lower right
                corner[2] = left[left.size() - 1];   // back, upper left
                corner[3] = right[right.size() - 1]; // back, upper right
                corner[4] = left[0] + frontPlane;     // front, lower left
                corner[5] = right[0] + frontPlane;    // front, lower right
                corner[6] = left[left.size() - 1] + frontPlane;   // front, upper left
                corner[7] = right[right.size() - 1] + frontPlane; // front, upper right
                int maxCornerNeighbors = (nz < 1) ? 4 : 8;
                conduit::index_t z0 = (nz < 1) ? 0 : -1;
                conduit::index_t z1 = (nz < 1) ? 0 : 1;
                conduit::index_t neighborId = InvalidDomain;
                const conduit::index_t cornerNeighbors[][3] = {
                    {-1, -1, z0},
                    {1, -1, z0},
                    {-1, 1, z0},
                    {1, 1, z0},
                    {-1, -1, z1},
                    {1, -1, z1},
                    {-1, 1, z1},
                    {1, 1, z1}
                };
                for(int ni = 0; ni < maxCornerNeighbors; ni++)
                {
                    if((neighborId = domainIndex(cornerNeighbors[ni], domain, domains)) != InvalidDomain)
                    {
                        auto name = adjset_name(thisDom, neighborId);
                        conduit::Node &group = groups[name];
                        group["neighbors"] = neighborId;
                        group["values"] = reorder ? old2NewPoint[corner[ni]] : corner[ni];
                    }    
                }

                // Make edge neighbors for 3D
                //-----------------------------------------------------------
                if(nz > 0)
                {
                    // Back, left edge.
                    if((neighborId = domainIndex(-1, 0, -1, domain, domains)) != InvalidDomain)
                    {
                        auto name = adjset_name(thisDom, neighborId);
                        conduit::Node &group = groups[name];
                        group["neighbors"] = neighborId;
                        addAdjsetValues(left, 0, reorder, old2NewPoint, group["values"]);
                    }
                    // Back, right edge.
                    if((neighborId = domainIndex(1, 0, -1, domain, domains)) != InvalidDomain)
                    {
                        auto name = adjset_name(thisDom, neighborId);
                        conduit::Node &group = groups[name];
                        group["neighbors"] = neighborId;
                        addAdjsetValues(right, 0, reorder, old2NewPoint, group["values"]);
                    }
                    // Back, bottom edge.
                    if((neighborId = domainIndex(0, -1, -1, domain, domains)) != InvalidDomain)
                    {
                        auto name = adjset_name(thisDom, neighborId);
                        conduit::Node &group = groups[name];
                        group["neighbors"] = neighborId;
                        addAdjsetValues(bottom, 0, reorder, old2NewPoint, group["values"]);
                    }
                    // Back, top edge.
                    if((neighborId = domainIndex(0, 1, -1, domain, domains)) != InvalidDomain)
                    {
                        auto name = adjset_name(thisDom, neighborId);
                        conduit::Node &group = groups[name];
                        group["neighbors"] = neighborId;
                        addAdjsetValues(top, 0, reorder, old2NewPoint, group["values"]);
                    }
                    // Lower left edge
                    if((neighborId = domainIndex(-1, -1, 0, domain, domains)) != InvalidDomain)
                    {
                        auto name = adjset_name(thisDom, neighborId);
                        std::vector<conduit::index_t> values;
                        values.reserve(nz);
                        for(conduit::index_t zi = 0; zi <= nz; zi++)
                            values.push_back(corner[0] + zi * ptsPerPlane);
                        conduit::Node &group = groups[name];
                        group["neighbors"] = neighborId;
                        addAdjsetValues(values, 0, reorder, old2NewPoint, group["values"]);
                    }
                    // Lower right edge
                    if((neighborId = domainIndex(1, -1, 0, domain, domains)) != InvalidDomain)
                    {
                        auto name = adjset_name(thisDom, neighborId);
                        std::vector<conduit::index_t> values;
                        values.reserve(nz);
                        for(conduit::index_t zi = 0; zi <= nz; zi++)
                            values.push_back(corner[1] + zi * ptsPerPlane);
                        conduit::Node &group = groups[name];
                        group["neighbors"] = neighborId;
                        addAdjsetValues(values, 0, reorder, old2NewPoint, group["values"]);
                    }
                    // Upper left edge
                    if((neighborId = domainIndex(-1, 1, 0, domain, domains)) != InvalidDomain)
                    {
                        auto name = adjset_name(thisDom, neighborId);
                        std::vector<conduit::index_t> values;
                        values.reserve(nz);
                        for(conduit::index_t zi = 0; zi <= nz; zi++)
                            values.push_back(corner[2] + zi * ptsPerPlane);
                        conduit::Node &group = groups[name];
                        group["neighbors"] = neighborId;
                        addAdjsetValues(values, 0, reorder, old2NewPoint, group["values"]);
                    }
                    // Upper right edge
                    if((neighborId = domainIndex(1, 1, 0, domain, domains)) != InvalidDomain)
                    {
                        auto name = adjset_name(thisDom, neighborId);
                        std::vector<conduit::index_t> values;
                        values.reserve(nz);
                        for(conduit::index_t zi = 0; zi <= nz; zi++)
                            values.push_back(corner[3] + zi * ptsPerPlane);
                        conduit::Node &group = groups[name];
                        group["neighbors"] = neighborId;
                        addAdjsetValues(values, 0, reorder, old2NewPoint, group["values"]);
                    }
                    // Front, left edge.
                    if((neighborId = domainIndex(-1, 0, 1, domain, domains)) != InvalidDomain)
                    {
                        auto name = adjset_name(thisDom, neighborId);
                        conduit::Node &group = groups[name];
                        group["neighbors"] = neighborId;
                        addAdjsetValues(left, frontPlane, reorder, old2NewPoint, group["values"]);
                    }
                    // Front, right edge.
                    if((neighborId = domainIndex(1, 0, 1, domain, domains)) != InvalidDomain)
                    {
                        auto name = adjset_name(thisDom, neighborId);
                        conduit::Node &group = groups[name];
                        group["neighbors"] = neighborId;
                        addAdjsetValues(right, frontPlane, reorder, old2NewPoint, group["values"]);
                    }
                    // Front, bottom edge.
                    if((neighborId = domainIndex(0, -1, 1, domain, domains)) != InvalidDomain)
                    {
                        auto name = adjset_name(thisDom, neighborId);
                        conduit::Node &group = groups[name];
                        group["neighbors"] = neighborId;
                        addAdjsetValues(bottom, frontPlane, reorder, old2NewPoint, group["values"]);
                    }
                    // Front, top edge.
                    if((neighborId = domainIndex(0, 1, 1, domain, domains)) != InvalidDomain)
                    {
                        auto name = adjset_name(thisDom, neighborId);
                        conduit::Node &group = groups[name];
                        group["neighbors"] = neighborId;
                        addAdjsetValues(top, frontPlane, reorder, old2NewPoint, group["values"]);
                    }
                }

                // Make "face" neighbors.
                //-----------------------------------------------------------
                conduit::index_t neighbor[6];
                neighbor[BoundaryLeft]   = domainIndex(-1, 0, 0, domain, domains);
                neighbor[BoundaryRight]  = domainIndex(1, 0, 0, domain, domains);
                neighbor[BoundaryBottom] = domainIndex(0, -1, 0, domain, domains);
                neighbor[BoundaryTop]    = domainIndex(0, 1, 0, domain, domains);
                neighbor[BoundaryBack]   = domainIndex(0, 0, -1, domain, domains);
                neighbor[BoundaryFront]  = domainIndex(0, 0, 1, domain, domains);
                int maxNeighbors = (nz < 1) ? 4 : 6;
                for(int ni = 0; ni < maxNeighbors; ni++)
                {
                    // If this domain has no neighbor in the current direction, skip.
                    if(neighbor[ni] == InvalidDomain)
                         continue;

                    // Iterate over faces and come up with unique points.
                    bool flags[6] = {false, false, false, false, false, false};
                    flags[ni] = true;
                    unique.clear();
                    ptvec.clear();
                    if(nz < 1)
                        iterateBoundary2D(tiles, nx, ny, flags, addPoints);
                    else
                        iterateBoundary3D(tiles, nx, ny, nz, ptsPerPlane, flags, addPoints);

                    if(!ptvec.empty())
                    {
                        auto name = adjset_name(thisDom, neighbor[ni]);
                        conduit::Node &group = groups[name];
                        group["neighbors"] = neighbor[ni];
                        addAdjsetValues(ptvec, 0, reorder, old2NewPoint, group["values"]);
                    }
                } // end for
            }
        }
    }
}

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::examples::detail --
//-----------------------------------------------------------------------------

void
tiled(conduit::index_t nx, conduit::index_t ny, conduit::index_t nz,
      conduit::Node &res, const conduit::Node &options)
{
    detail::Tiler T;
    T.generate(nx, ny, nz, res, options);
}

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh::examples --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::mesh --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint --
//-----------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------
// -- end conduit --
//-----------------------------------------------------------------------------

#endif



