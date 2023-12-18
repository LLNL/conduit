// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_curves.hpp
///
//-----------------------------------------------------------------------------

#ifndef CONDUIT_BLUEPRINT_CURVES_HPP
#define CONDUIT_BLUEPRINT_CURVES_HPP

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
// -- begin conduit::blueprint::curves --
//-----------------------------------------------------------------------------
namespace curves
{

//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::curves::detail --
//-----------------------------------------------------------------------------
namespace detail
{

// NOTE: This code is based on Python code by Jakub Cervený from
//       https://github.com/jakubcerveny/gilbert
//
// The code was ported to C++ for use in Conduit.
//
// The original code's license follows:

/*
BSD 2-Clause License

Copyright (c) 2018, Jakub Cervený
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

//----------------------------------------------------------------------------
template <typename Precision>
inline Precision sgn(Precision x)
{
    return (x < 0) ? -1 : ((x > 0) ? 1 : 0);
}

//----------------------------------------------------------------------------
template <typename Precision>
inline Precision iabs(Precision x)
{
    return (x < 0) ? -x : x;
}

//----------------------------------------------------------------------------
template <typename IndexType, typename Consumer>
void
generate2d(IndexType x, IndexType y,
    IndexType ax, IndexType ay,
    IndexType bx, IndexType by,
    Consumer &&consumer)
{
    auto w = iabs(ax + ay);
    auto h = iabs(bx + by);
    // unit major direction
    auto dax = sgn(ax);
    auto day = sgn(ay);
    // unit orthogonal direction
    auto dbx = sgn(bx);
    auto dby = sgn(by);

    if(h == 1)
    {
        // trivial row fill
        for(IndexType i = 0; i < w; i++)
        {
            consumer(x, y);

            x += dax;
            y += day;
        }
        return;
    }

    if(w == 1)
    {
        // trivial column fill
        for(IndexType i = 0; i < h; i++)
        {
            consumer(x, y);

            x += dbx;
            y += dby;
        }
        return;
    }

    auto ax2 = ax / 2;
    auto ay2 = ay / 2;
    auto bx2 = bx / 2;
    auto by2 = by / 2;

    auto w2 = iabs(ax2 + ay2);
    auto h2 = iabs(bx2 + by2);

    if(2*w > 3*h)
    {
        if((w2 % 2) && (w > 2))
        {
            // prefer even steps
            ax2 += dax;
            ay2 += day;
        }
        // long case: split in two parts only
        generate2d(x, y, ax2, ay2, bx, by, consumer);
        generate2d(x+ax2, y+ay2, ax-ax2, ay-ay2, bx, by, consumer);
    }
    else
    {
        if((h2 % 2) && (h > 2))
        {
            // prefer even steps
            bx2 += dbx;
            by2 += dby;
        }

        // standard case: one step up, one long horizontal, one step down
        generate2d(x, y, bx2, by2, ax2, ay2, consumer);
        generate2d(x+bx2, y+by2, ax, ay, bx-bx2, by-by2, consumer);
        generate2d(x+(ax-dax)+(bx2-dbx), y+(ay-day)+(by2-dby),
                              -bx2, -by2, -(ax-ax2), -(ay-ay2), consumer);
    }
}

//----------------------------------------------------------------------------
template <typename IndexType, typename Consumer>
void
generate3d(IndexType x, IndexType y, IndexType z,
           IndexType ax, IndexType ay, IndexType az,
           IndexType bx, IndexType by, IndexType bz,
           IndexType cx, IndexType cy, IndexType cz,
           Consumer &&consumer)
{
    auto w = iabs(ax + ay + az);
    auto h = iabs(bx + by + bz);
    auto d = iabs(cx + cy + cz);

    // unit major direction ("right")
    auto dax = sgn(ax);
    auto day = sgn(ay);
    auto daz = sgn(az);
    // unit ortho direction ("forward")
    auto dbx = sgn(bx);
    auto dby = sgn(by);
    auto dbz = sgn(bz);
    // unit ortho direction ("up")
    auto dcx = sgn(cx);
    auto dcy = sgn(cy);
    auto dcz = sgn(cz);

    // trivial row/column fills
    if(h == 1 && d == 1)
    {
        for(IndexType i = 0; i < w; i++)
        {
            consumer(x, y, z);

            x += dax;
            y += day;
            z += daz;
        }
        return;
    }

    if(w == 1 && d == 1)
    {
        for(IndexType i = 0; i < h; i++)
        {
            consumer(x, y, z);

            x += dbx;
            y += dby;
            z += dbz;
        }
        return;
    }

    if(w == 1 && h == 1)
    {
        for(IndexType i = 0; i < d; i++)
        {
            consumer(x, y, z);

            x += dcx;
            y += dcy;
            z += dcz;
        }
        return;
    }

    auto ax2 = ax / 2;
    auto ay2 = ay / 2;
    auto az2 = az / 2;

    auto bx2 = bx / 2;
    auto by2 = by / 2;
    auto bz2 = bz / 2;

    auto cx2 = cx / 2;
    auto cy2 = cy / 2;
    auto cz2 = cz / 2;

    auto w2 = iabs(ax2 + ay2 + az2);
    auto h2 = iabs(bx2 + by2 + bz2);
    auto d2 = iabs(cx2 + cy2 + cz2);

    // prefer even steps
    if((w2 % 2) && (w > 2))
    {
        ax2 += dax;
        ay2 += day;
        az2 += daz;
    }
    if((h2 % 2) && (h > 2))
    {
        bx2 += dbx;
        by2 += dby;
        bz2 += dbz;
    }
    if((d2 % 2) && (d > 2))
    {
        cx2 += dcx;
        cy2 += dcy;
        cz2 += dcz;
    }

    // wide case, split in w only
    if((2*w > 3*h) && (2*w > 3*d))
    {
        generate3d(x, y, z,
                   ax2, ay2, az2,
                   bx, by, bz,
                   cx, cy, cz, consumer);

        generate3d(x+ax2, y+ay2, z+az2,
                   ax-ax2, ay-ay2, az-az2,
                   bx, by, bz,
                   cx, cy, cz, consumer);
    }
    // do not split in d
    else if(3*h > 4*d)
    {
        generate3d(x, y, z,
                   bx2, by2, bz2,
                   cx, cy, cz,
                   ax2, ay2, az2, consumer);

        generate3d(x+bx2, y+by2, z+bz2,
                   ax, ay, az,
                   bx-bx2, by-by2, bz-bz2,
                   cx, cy, cz, consumer);

        generate3d(x+(ax-dax)+(bx2-dbx),
                   y+(ay-day)+(by2-dby),
                   z+(az-daz)+(bz2-dbz),
                   -bx2, -by2, -bz2,
                   cx, cy, cz,
                   -(ax-ax2), -(ay-ay2), -(az-az2), consumer);
    }
    // do not split in h
    else if(3*d > 4*h)
    {
        generate3d(x, y, z,
                   cx2, cy2, cz2,
                   ax2, ay2, az2,
                   bx, by, bz, consumer);

        generate3d(x+cx2, y+cy2, z+cz2,
                   ax, ay, az,
                   bx, by, bz,
                   cx-cx2, cy-cy2, cz-cz2, consumer);

        generate3d(x+(ax-dax)+(cx2-dcx),
                   y+(ay-day)+(cy2-dcy),
                   z+(az-daz)+(cz2-dcz),
                   -cx2, -cy2, -cz2,
                   -(ax-ax2), -(ay-ay2), -(az-az2),
                   bx, by, bz, consumer);
    }
    // regular case, split in all w/h/d
    else
    {
        generate3d(x, y, z,
                   bx2, by2, bz2,
                   cx2, cy2, cz2,
                   ax2, ay2, az2, consumer);

        generate3d(x+bx2, y+by2, z+bz2,
                   cx, cy, cz,
                   ax2, ay2, az2,
                   bx-bx2, by-by2, bz-bz2, consumer);

        generate3d(x+(bx2-dbx)+(cx-dcx),
                   y+(by2-dby)+(cy-dcy),
                   z+(bz2-dbz)+(cz-dcz),
                   ax, ay, az,
                   -bx2, -by2, -bz2,
                   -(cx-cx2), -(cy-cy2), -(cz-cz2), consumer);

        generate3d(x+(ax-dax)+bx2+(cx-dcx),
                   y+(ay-day)+by2+(cy-dcy),
                   z+(az-daz)+bz2+(cz-dcz),
                   -cx, -cy, -cz,
                   -(ax-ax2), -(ay-ay2), -(az-az2),
                   bx-bx2, by-by2, bz-bz2, consumer);

        generate3d(x+(ax-dax)+(bx2-dbx),
                   y+(ay-day)+(by2-dby),
                   z+(az-daz)+(bz2-dbz),
                   -bx2, -by2, -bz2,
                   cx2, cy2, cz2,
                   -(ax-ax2), -(ay-ay2), -(az-az2), consumer);
    }
}

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::curves::detail --
//-----------------------------------------------------------------------------

//---------------------------------------------------------------------------
/**
 @brief Generalized Hilbert ('gilbert') space-filling curve for arbitrary-sized
        2D rectangular grids. Generates discrete 2D coordinates to fill a rectangle
        of size (width x height).

 @param width The width of the block.
 @param height The height of the block.
 @param consumer A consumer that will take the I,J indices of the current
                 zone as the code winds through the block.
 */
template <typename IndexType, typename Consumer>
void gilbert2d(IndexType width, IndexType height, Consumer &&consumer)
{
    IndexType O{0};
    if(width >= height)
        detail::generate2d(O, O, width, O, O, height, consumer);
    else
        detail::generate2d(O, O, O, height, width, O, consumer);
}

//---------------------------------------------------------------------------
/**
 @brief Generalized Hilbert ('Gilbert') space-filling curve for arbitrary-sized
        3D rectangular grids. Generates discrete 3D coordinates to fill a cuboid
        of size (width x height x depth). Even sizes are recommended in 3D.

 @param width The width of the block.
 @param height The height of the block.
 @param depth The depth of the block.
 @param consumer A consumer that will take the I,J indices of the current
                 zone as the code winds through the block.
 */
template <typename IndexType, typename Consumer>
void gilbert3d(IndexType width, IndexType height, IndexType depth, Consumer &&consumer)
{
    IndexType O{0};
    if(width >= height && width >= depth)
    {
        detail::generate3d(O, O, O,
                           width, O, O,
                           O, height, O,
                           O, O, depth,
                           consumer);
    }
    else if(height >= width && height >= depth)
    {
        detail::generate3d(O, O, O,
                           O, height, O,
                           width, O, O,
                           O, O, depth,
                           consumer);
    }
    else // depth >= width and depth >= height
    {
        detail::generate3d(O, O, O,
                           O, O, depth,
                           width, O, O,
                           O, height, O,
                           consumer);
    }
}

}
//-----------------------------------------------------------------------------
// -- end conduit::blueprint::curves --
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
