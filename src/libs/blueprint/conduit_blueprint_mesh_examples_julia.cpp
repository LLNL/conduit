// Copyright (c) Lawrence Livermore National Security, LLC and other Conduit
// Project developers. See top-level LICENSE AND COPYRIGHT files for dates and
// other details. No copyright assignment is required to contribute to Conduit.

//-----------------------------------------------------------------------------
///
/// file: conduit_blueprint_mesh_examples_julia.cpp
///
//-----------------------------------------------------------------------------

#if defined(CONDUIT_PLATFORM_WINDOWS)
#define NOMINMAX
#undef min
#undef max
#include "windows.h"
#endif

//-----------------------------------------------------------------------------
// std lib includes
//-----------------------------------------------------------------------------
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <cassert>
#include <map>
#include <set>
#include <vector>
#include <queue>

//-----------------------------------------------------------------------------
// conduit includes
//-----------------------------------------------------------------------------
#include "conduit_blueprint_mesh_examples.hpp"
#include "conduit_blueprint_mesh.hpp"


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
// -- begin conduit::blueprint::mesh --
//-----------------------------------------------------------------------------
namespace mesh
{


//-----------------------------------------------------------------------------
// -- begin conduit::blueprint::mesh::examples --
//-----------------------------------------------------------------------------
namespace examples
{

void julia_fill_values(index_t nx,
                       index_t ny,
                       float64 x_min,
                       float64 x_max,
                       float64 y_min,
                       float64 y_max,
                       float64 c_re,
                       float64 c_im,
                       int32_array &out)
{
    index_t idx = 0;
    for(index_t j = 0; j < ny; j++)
    {
        for(index_t i = 0; i < nx; i++)
        {
            float64 zx = float64(i) / float64(nx-1);
            float64 zy = float64(j) / float64(ny-1);

            zx = x_min + (x_max - x_min) * zx;
            zy = y_min + (y_max - y_min) * zy;

            int32 iter = 0;
            int32 max_iter = 1000;

            while( (zx * zx) + (zy * zy ) < 4.0 && iter < max_iter)
            {
                float64 x_temp = zx*zx - zy*zy;
                zy = 2*zx*zy  + c_im;
                zx = x_temp    + c_re;
                iter++;
            }
            if(iter == max_iter)
            {
                out[idx] = 0;
            }
            else
            {
                out[idx] = iter;
            }

            idx++;
        }
    }
}
//---------------------------------------------------------------------------//

void paint_2d_nestsets(conduit::Node &domain,
                       const std::string topo_name)
{

  if(!domain.has_path("topologies/"+topo_name))
  {
    CONDUIT_ERROR("Paint nestsets: no topology named: "<<topo_name);
  }

  const Node &topo = domain["topologies/"+topo_name];

  if(topo["type"].as_string() == "unstructured")
  {
    CONDUIT_ERROR("Paint nestsets: cannot paint on unstructured topology");
  }

  int el_dims[2] = {1,1};
  if(topo["type"].as_string() == "structured")
  {
    el_dims[0] = topo["elements/dims/i"].to_int32();
    el_dims[1] = topo["elements/dims/j"].to_int32();
  }
  else
  {
    const std::string coord_name = topo["coordset"].as_string();
    const Node &coords = domain["coordsets/"+coord_name];
    if(coords["type"].as_string() == "uniform")
    {
      el_dims[0] = coords["dims/i"].as_int32() - 1;
      el_dims[1] = coords["dims/j"].as_int32() - 1;
    }
    else if(coords["type"].as_string() == "rectilinear")
    {
      el_dims[0] = (int)(coords["values/x"].dtype().number_of_elements() - 1);
      el_dims[1] = (int)(coords["values/y"].dtype().number_of_elements() - 1);
    }
    else
    {
      CONDUIT_ERROR("unknown coord type");
    }
  }

  const int32 field_size = el_dims[0] * el_dims[1];

  Node &levels_field = domain["fields/mask"];
  levels_field["association"] = "element";
  levels_field["topology"] = topo_name;
  levels_field["values"] = DataType::int32(field_size);
  int32_array levels = levels_field["values"].value();


  for(int i = 0; i < field_size; ++i)
  {
    levels[i] = 0;
  }

  int nest_id = -1;
  for(int i = 0; i < domain["nestsets"].number_of_children(); ++i)
  {
    const Node &nestset = domain["nestsets"].child(0);
    if(nestset["topology"].as_string() == topo_name)
    {
      nest_id = i;
      break;
    }
  }
  if(nest_id == -1) return;

  const Node &nestset = domain["nestsets"].child(nest_id);
  const index_t windows = nestset["windows"].number_of_children();
  for(index_t i = 0; i < windows; ++i)
  {
    const Node &window = nestset["windows"].child(i);
    if(window["domain_type"].as_string() != "child")
    {
      continue;
    }

    int origin[2];
    origin[0] = window["origin/i"].to_int32();
    origin[1] = window["origin/j"].to_int32();

    int dims[2];
    dims[0] = window["dims/i"].to_int32();
    dims[1] = window["dims/j"].to_int32();
    // all the nesting relationship is local
    for(int y = origin[1]; y < origin[1] + dims[1]; ++y)
    {
      const int32 y_offset = y * el_dims[0];
      for(int x = origin[0]; x < origin[0] + dims[0]; ++x)
      {
        levels[y_offset + x] += 1;
      }
    }
  }
}
//---------------------------------------------------------------------------//

void gap_scanner(const std::vector<int32> &values,
                 const index_t start,
                 const index_t end,
                 const index_t offset,
                 int32 gap[2])
{
  bool in_gap = false;
  int32 gap_length = 0;
  gap[0] = -1; // index of gap
  gap[1] = 0;  // length of gap
  for(index_t i = start - offset; i <= end - offset; ++i)
  {
    if(values[i] == 0)
    {
      if(in_gap) gap_length++;
      else
      {
        gap_length = 1;
        in_gap = true;
      }
    }
    else
    {
      if(in_gap)
      {
        if(gap_length > gap[1])
        {
          gap[0] = (int32)(i + offset);
          gap[1] = gap_length;
        }
        in_gap = false;
      }
    }
  }
}
//---------------------------------------------------------------------------//

void inflection_scanner(const std::vector<int32> &values,
                        const index_t start,
                        const index_t end,
                        const index_t offset,
                        int32 crit[2])
{
  crit[0] = -1;
  crit[1] = 0;
  int32 prev = 0;
  for(index_t i = start + 1 - offset; i <= end - 1 - offset; ++i)
  {
    // second derivitive using finite differences
    int32 deriv = values[i + 1] - 2 * values[i] + values[i-1];
    // inflection point
    if((prev < 0 && deriv > 0) || (prev > 0 && deriv < 0))
    {
      int32 mag = abs(deriv - prev);
      if(mag  > crit[1])
      {
        crit[0] = (int32)(i + offset);
        crit[1] = mag;
      }
    }
    prev = deriv;
  }
}

struct AABB
{
  index_t box[3][2];

  AABB()
  {
    box[0][0] = std::numeric_limits<index_t>::max();
    box[0][1] = std::numeric_limits<index_t>::min();
    box[1][0] = std::numeric_limits<index_t>::max();
    box[1][1] = std::numeric_limits<index_t>::min();
    box[2][0] = std::numeric_limits<index_t>::max();
    box[2][1] = std::numeric_limits<index_t>::min();
  }

  bool valid(index_t axis)
  {
    return box[axis][0] <= box[axis][1];
  }

  index_t size()
  {
    index_t res = 0;
    res = (box[0][1] - box[0][0] + 1) * (box[1][1] - box[1][0] + 1);
    if(valid(2))
    {
      res *= box[2][1] - box[2][0] + 1;
    }
    return res;
  }

  void print()
  {

    std::cout<<"size "<<size()<<" "
              <<"("<<box[0][0]<<","<<box[0][1]<<") - "
              <<"("<<box[1][0]<<","<<box[1][0]<<")\n";
  }

  index_t length(index_t axis)
  {
    return (box[axis][1] - box[axis][0] + 1);
  }

  index_t min(index_t axis)
  {
    return box[axis][0];
  }

  index_t max(index_t axis)
  {
    return box[axis][1];
  }

  void split(const index_t axis,
             const index_t index,
             AABB &left,
             AABB &right)
  {
    for(index_t i = 0; i < 3; ++i)
    {
      if(i == axis)
      {
        left.box[i][0] = box[i][0];
        left.box[i][1] = index;
        right.box[i][0] = index + 1;
        right.box[i][1] = box[i][1];
      }
      else
      {
        left.box[i][0] = box[i][0];
        left.box[i][1] = box[i][1];
        right.box[i][0] = box[i][0];
        right.box[i][1] = box[i][1];
      }
    }
  }

  void include(int axis, index_t pos)
  {
    box[axis][0] = std::min(pos, box[axis][0]);
    box[axis][1] = std::max(pos, box[axis][1]);
  }

  void mid_split(AABB &left, AABB &right)
  {
    index_t axis = 0;
    index_t len = 0;
    for(index_t i = 0; i < 3; i++)
    {
      if(valid(i))
      {
        index_t size = length(i);
        if(size > len)
        {
          axis  = i;
          len = size;
        }
      }
    }

    index_t pos = len/2 + box[axis][0] - 1;
    split(axis, pos, left, right);
  }

};
//---------------------------------------------------------------------------//

// creates a vector of sub boxs using a list of
// flags that marks 'interesting' zones
void
sub_boxs(const std::vector<int32> &flags,
         const index_t nx,
         const index_t ny,
         const float64 efficiency,
         const int32 min_size,
         std::vector<AABB> &refined)
{
  AABB mesh_box;
  mesh_box.include(0,0);
  mesh_box.include(0,nx-1);
  mesh_box.include(1,0);
  mesh_box.include(1,ny-1);

  std::queue<AABB> aabbs;
  aabbs.push(mesh_box);

  while(!aabbs.empty())
  {
    AABB current = aabbs.front();
    index_t dx = current.length(0);
    index_t dy = current.length(1);
    std::vector<int32> x_bins(dx, 0);
    std::vector<int32> y_bins(dy, 0);

    int32 flag_count = 0;
    AABB aabb;
    // find the tight AABB containing flags
    for(index_t y = current.min(1); y <= current.max(1); ++y)
    {
      for(index_t x = current.min(0); x <= current.max(0); ++x)
      {
        index_t offset = y * nx + x;
        int32 flag = flags[offset];
        if(flag == 1)
        {
          aabb.include(0, x);
          aabb.include(1, y);
        }
        x_bins[x - current.min(0)] += flag;
        y_bins[y - current.min(1)] += flag;
        flag_count += flag;
      }
    }

    // terminating conditions
    if(flag_count == 0)
    {
      aabbs.pop();
      continue;
    }

    index_t subsize = aabb.size();
    float64 ratio = float64(flag_count) / float64(subsize);

    if(ratio > efficiency || subsize < min_size)
    {
      refined.push_back(aabb);
      aabbs.pop();
      continue;
    }

    // find a split

    // look for the longest gap that divides two 'clusters' of flags
    // this is the best kind of split
    int32 x_gap[2];
    int32 y_gap[2];
    gap_scanner(x_bins, aabb.min(0), aabb.max(0), current.min(0), x_gap);
    gap_scanner(y_bins, aabb.min(1), aabb.max(1), current.min(1), y_gap);
    if((x_gap[0] != -1 || y_gap[0] != -1) &&
        x_gap[0] != aabb.min(0) &&
        x_gap[0] != aabb.max(0) &&
        y_gap[0] != aabb.min(1) &&
        y_gap[0] != aabb.max(1))
    {
      AABB left, right;
      if(x_gap[1] > y_gap[1])
      {
        aabb.split(0, x_gap[0], left, right);
      }
      else
      {
        aabb.split(1, y_gap[0], left, right);
      }
      aabbs.pop();
      aabbs.push(left);
      aabbs.push(right);
      continue;
    }

    // look for splits defined by zero crossovers of the
    // second derivitive.
    int32 x_crit[2];
    int32 y_crit[2];
    inflection_scanner(x_bins, aabb.min(0), aabb.max(0), current.min(0),x_crit);
    inflection_scanner(y_bins, aabb.min(1), aabb.max(1), current.min(1),y_crit);

    if((x_crit[0] != -1 || y_crit[0] != -1) &&
        x_crit[0] != aabb.min(0) &&
        x_crit[0] != aabb.max(0) &&
        y_crit[0] != aabb.min(1) &&
        y_crit[0] != aabb.max(1))
    {
      AABB left, right;
      if(x_crit[1] > y_crit[1])
      {
        aabb.split(0, x_crit[0], left, right);
      }
      else
      {
        aabb.split(1, y_crit[0], left, right);
      }

      aabbs.pop();
      aabbs.push(left);
      aabbs.push(right);
      continue;
    }

    // if we are here then gaps and inflection failed
    // so split on the longest axis
    AABB left, right;
    aabb.mid_split(left, right);

    aabbs.pop();
    aabbs.push(left);
    aabbs.push(right);
  }
}



//---------------------------------------------------------------------------//
void julia(index_t nx,
           index_t ny,
           float64 x_min,
           float64 x_max,
           float64 y_min,
           float64 y_max,
           float64 c_re,
           float64 c_im,
           Node &res)
{
    res.reset();
    // create a rectilinear coordset
    res["coordsets/coords/type"] = "rectilinear";
    res["coordsets/coords/values/x"] = DataType::float64(nx+1);
    res["coordsets/coords/values/y"] = DataType::float64(ny+1);

    float64_array x_coords = res["coordsets/coords/values/x"].value();
    float64_array y_coords = res["coordsets/coords/values/y"].value();

    float64 dx = (x_max - x_min) / float64(nx);
    float64 dy = (y_max - y_min) / float64(ny);

    float64 vx = x_min;
    for(index_t i =0; i< nx+1; i++)
    {
        x_coords[i] = vx;
        vx+=dx;
    }

    float64 vy = y_min;
    for(index_t i =0; i< ny+1; i++)
    {
        y_coords[i] = vy;
        vy+=dy;
    }

    // create the topology

    res["topologies/topo/type"] = "rectilinear";
    res["topologies/topo/coordset"] = "coords";


    // create the fields

    res["fields/iters/association"] = "element";
    res["fields/iters/topology"] = "topo";
    res["fields/iters/values"] = DataType::int32(nx * ny);

    int32_array out = res["fields/iters/values"].value();

    julia_fill_values(nx,ny,
                      x_min, x_max,
                      y_min, y_max,
                      c_re, c_im,
                      out);
}

//---------------------------------------------------------------------------//

int32 refine(int32 domain_index,
             int32 domain_id_start,
             float64 threshold,
             float64 efficiency,
             int32 min_size,
             float64 c_re,
             float64 c_im,
             Node &res)
{
  Node &domain = res.child(domain_index);
  domain["nestsets/nest/association"] = "element";
  domain["nestsets/nest/topology"] = "topo";

  index_t nx = domain["coordsets/coords/values/x"].dtype().number_of_elements() - 1;
  index_t ny = domain["coordsets/coords/values/y"].dtype().number_of_elements() - 1;
  float64_array x_coords = domain["coordsets/coords/values/x"].value();
  float64_array y_coords = domain["coordsets/coords/values/y"].value();

  int32_array iters = domain["fields/iters/values"].value();

  std::vector<int32> flags(nx*ny);
  std::vector<int32> der(nx*ny);

  // perform a 2d stencil and calculate the mag
  // of the second derivitive of iters using
  // central differences
  for(index_t i = 0; i < nx*ny; ++i)
  {
    int32 flag = 0;
    index_t x = i % nx;
    index_t y = i / nx;
    float32 x_vals[3];
    float32 y_vals[3];
    for(index_t o = 0; o < 3; ++o)
    {
      index_t x_o = std::min(std::max(x+o-1,index_t(0)),nx-1);
      index_t y_o = std::min(std::max(y+o-1,index_t(0)),ny-1);
      x_vals[o] = float32(iters[y * nx + x_o]);
      y_vals[o] = float32(iters[y_o * nx + x]);
    }
    float32 ddx = std::abs(x_vals[0] - 2.f * x_vals[1] + x_vals[2]);
    float32 ddy = std::abs(y_vals[0] - 2.f * y_vals[1] + y_vals[2]);
    float32 eps = sqrt(ddx*ddx + ddy*ddy);
    // TODO, should der be a floating point # here?
    der[i] = eps;
    if(eps > threshold)
    {
      flag = 1;
    }
    flags[i] = flag;
  }

  // split the current domain based on the
  // flags
  std::vector<AABB> boxs;
  sub_boxs(flags, nx, ny, efficiency, min_size, boxs);

  // create, fill and update nestsets of refined domains
  int domain_id = domain_id_start;
  for(size_t i = 0; i < boxs.size(); ++i)
  {
    // create the current domain
    std::ostringstream oss;
    oss << "domain_" << std::setw(6) << std::setfill('0') << domain_id;
    std::string domain_name = oss.str();
    domain_id++;

    Node &child = res[domain_name];
    AABB aabb = boxs[i];

    index_t cnx = aabb.length(0);
    index_t cny = aabb.length(1);
    float64 cx_min = x_coords[aabb.min(0)];
    float64 cx_max = x_coords[aabb.max(0)+1];

    float64 cy_min = y_coords[aabb.min(1)];
    float64 cy_max = y_coords[aabb.max(1)+1];

    julia(cnx * 2,
          cny * 2,
          cx_min,
          cx_max,
          cy_min,
          cy_max,
          c_re,
          c_im,
          child);

     child["nestsets/nest/association"] = "element";
     child["nestsets/nest/topology"] = "topo";

     std::string window1, window2;
     oss.str("");
     oss << "window_" <<domain_index << "_" << domain_id;
     window1 = oss.str();
     oss.str("");
     oss << "window_" <<domain_id<< "_" << domain_index;
     window2 = oss.str();

     Node &pwindow = domain["nestsets/nest/windows/"+window1];
     pwindow["domain_id"] = domain_id;
     pwindow["domain_type"] = "child";
     pwindow["origin/i"] = aabb.min(0);
     pwindow["origin/j"] = aabb.min(1);
     pwindow["dims/i"] = cnx;
     pwindow["dims/j"] = cny;
     pwindow["ratio/i"] = 2;
     pwindow["ratio/j"] = 2;

     child["nestsets/nest/association"] = "element";
     child["nestsets/nest/topology"] = "topo";
     Node &cwindow = child["nestsets/nest/windows/"+window2];
     cwindow["domain_id"] = domain_index;
     cwindow["domain_type"] = "parent";
     cwindow["origin/i"] = 0;
     cwindow["origin/j"] = 0;
     cwindow["dims/i"] = cnx * 2;
     cwindow["dims/j"] = cny * 2;
     cwindow["ratio/i"] = 2;
     cwindow["ratio/j"] = 2;
  }
  return (int32)boxs.size();
}

void julia_nestsets_complex(index_t nx,
                            index_t ny,
                            float64 x_min,
                            float64 x_max,
                            float64 y_min,
                            float64 y_max,
                            float64 c_re,
                            float64 c_im,
                            index_t levels,
                            Node &res)
{
  res.reset();
  // create the top level
  Node &parent = res["domain_000000"];
  julia(nx, ny, x_min, x_max, y_min, y_max, c_re, c_im, parent);

  // AMR knobs
  float64 threshold = 10.;   // 2nd derivitive flag threshold
  int32 min_size = 4;       // min num zones for refine
  float64 efficiency = .80; // target boxs count(flags)/size > effeciency

  int32 curr_domain = 0;
  int32 domain_count = 1;
  int32 children = 1;

  for(int32 i = 0; i < levels; ++i)
  {

    int32 level_count = 0;
    int32 offset = domain_count - children;
    for(int32 d = 0; d < children; ++d)
    {
      int32 count = refine(d + offset,
                           domain_count,
                           threshold,
                           efficiency,
                           min_size,
                           c_re,
                           c_im,
                           res);

      domain_count += count;
      level_count += count;
    }
    curr_domain += level_count;
    children = level_count;
    // for each level refinement threshold and min size
    threshold += 20;
    min_size *= 2;
  }

  // create a field on the mesh that flags zones that
  // are covered by a lower level of refinement
  for(int32 i = 0; i < res.number_of_children(); ++i)
  {
    paint_2d_nestsets(res.child(i), "topo");
  }

}

//---------------------------------------------------------------------------//
void julia_nestsets_simple(float64 x_min,
                           float64 x_max,
                           float64 y_min,
                           float64 y_max,
                           float64 c_re,
                           float64 c_im,
                           Node &res)
{
  res.reset();
  // create the top level
  Node &parent = res["domain_000000"];
  julia(8, 8, x_min, x_max, y_min, y_max, c_re, c_im, parent);

  float64_array x_coords = parent["coordsets/coords/values/x"].value();
  float64_array y_coords = parent["coordsets/coords/values/y"].value();

  float64 c_x_min= x_coords[2];
  float64 c_x_max= x_coords[6];
  float64 c_y_min= y_coords[2];
  float64 c_y_max= y_coords[6];
  Node &child = res["domain_000001"];
  julia(8, 8, c_x_min, c_x_max, c_y_min, c_y_max, c_re, c_im, child);

  parent["nestsets/nest/association"] = "element";
  parent["nestsets/nest/topology"] = "topo";

  Node &pwindow = parent["nestsets/nest/windows/window_0_1"];
  pwindow["domain_id"] = 1;
  pwindow["domain_type"] = "child";
  pwindow["origin/i"] = 2;
  pwindow["origin/j"] = 2;
  pwindow["dims/i"] = 4;
  pwindow["dims/j"] = 4;
  pwindow["ratio/i"] = 2;
  pwindow["ratio/j"] = 2;

  child["nestsets/nest/association"] = "element";
  child["nestsets/nest/topology"] = "topo";
  Node &cwindow = child["nestsets/nest/windows/window_1_0"];
  cwindow["domain_id"] = 0;
  cwindow["domain_type"] = "parent";
  cwindow["origin/i"] = 0;
  cwindow["origin/j"] = 0;
  cwindow["dims/i"] = 8;
  cwindow["dims/j"] = 8;
  cwindow["ratio/i"] = 2;
  cwindow["ratio/j"] = 2;

  for(int i = 0; i < res.number_of_children(); ++i)
  {
    paint_2d_nestsets(res.child(i), "topo");
  }

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
// -- end conduit::blueprint:: --
//-----------------------------------------------------------------------------


}
//-----------------------------------------------------------------------------
// -- end conduit:: --
//-----------------------------------------------------------------------------
