// Compile with: g++ -o main.exe main.cpp -Ofast -march=native -fno-omit-frame-pointer -fopenmp -unroll-loops

#include <iostream>
#include <vector>
#include <array>

#include <immintrin.h>

#include <random>
#include <chrono>
#include <algorithm>

// global variable to ensure that results do not get optimized away
extern volatile bool res = 0;

template<typename T>
struct Configuration{
  T x;
  T y;
  T z;
};

template<typename T>
struct ConfigurationOfArrays{
  std::array<T, 8> x;
  std::array<T, 8> y;
  std::array<T, 8> z;
};

template<typename T>
struct Sphere{
  T x;
  T y;
  T z;

  T r;
};


// Axis aligned box
template<typename T>
struct AABB{
  T xmin;
  T ymin;
  T zmin;

  T xmax;
  T ymax;
  T zmax;
};

template<typename T>
struct SphereArray{
  std::array<T, 8> x;
  std::array<T, 8> y;
  std::array<T, 8> z;

  T r;
};

template<typename T>
bool collides(const Sphere<T> &l, const Sphere<T> &r){
  const auto d = (l.x - r.x) * (l.x - r.x) + (l.y - r.y) * (l.y - r.y) + (l.z - r.z) * (l.z - r.z);
  if (d < (l.r + r.r) * (l.r + r.r)){
    return true;
  }

  return false;
}

template<typename T>
bool collides(const AABB<T> &aabb, const Sphere<T> &l){
  // copmute distance from center of sphere to box on each axis
  auto check = [&](
    const T pn,
    const T bmin,
    const T bmax ) -> T {
    const T v = pn;

    T out = 0;

    if ( v < bmin ){
      const T val = (bmin - v);
      out += val * val;
    }

    if ( v > bmax ){
      const T val = (v - bmax);
      out += val * val;
    }

    return out;
  };

  auto check_branchless = [&](
    const T pn,
    const T bmin,
    const T bmax ) -> T {
    const T lower_diff = std::max((bmin - pn), (T)0.);
    const T upper_diff = std::max((pn-bmax), (T)0.);
    const T out = std::max(
                          lower_diff*lower_diff,
                          upper_diff*upper_diff);
    return out;
  };

  // Squared distance
  T sq = 0.0;
  sq += check_branchless( l.x, aabb.xmin, aabb.xmax );
  sq += check_branchless( l.y, aabb.ymin, aabb.ymax );
  sq += check_branchless( l.z, aabb.zmin, aabb.zmax );

  return sq <= l.r * l.r;
}

bool simd_collides(const AABB<float> &aabb, const SphereArray<float> &rs){
  __m256 zero = _mm256_setzero_ps();

  // compare things, and get larger one
  // X
  // Computes
  //   std::max(
  //    std::max(xmin - x, 0)^2,
  //    std::max(x - xmax, 0)^2
  //   )
  auto check = [&](
    const float *p,
    const float bmin,
    const float bmax ) -> __m256 {
    __m256 xmin = _mm256_set1_ps(bmin); // broadcast xmin to packed float register
    __m256 xmax = _mm256_set1_ps(bmax); // broadcast xmax to packed float register
    __m256 xr = _mm256_load_ps(p); // load sphere x-pos

    __m256 ldx = _mm256_sub_ps(xmin, xr); // xmin - x
    __m256 udx = _mm256_sub_ps(xr, xmax); // x - xmax

    ldx =_mm256_max_ps(ldx, zero); // max(xmin-x, 0)
    udx =_mm256_max_ps(udx, zero); // max(x-xmax, 0)

    __m256 ldx2 = _mm256_mul_ps(ldx, ldx); // square the value
    __m256 udx2 = _mm256_mul_ps(udx, udx);

    __m256 xd =_mm256_max_ps(ldx2, udx2); // max(...)

    return xd;
  };

  __m256 xd = check(rs.x.data(), aabb.xmin, aabb.xmax);
  __m256 yd = check(rs.y.data(), aabb.ymin, aabb.ymax);
  __m256 zd = check(rs.z.data(), aabb.zmin, aabb.zmax);

  // compare final thingy
  __m256 d = _mm256_add_ps(xd, yd);
  d = _mm256_add_ps(d, zd);

  __m256 r2 = _mm256_set1_ps(rs.r*rs.r);
  __m256 cmp = _mm256_cmp_ps(d, r2, _CMP_LT_OQ);

  const int mask = _mm256_movemask_ps(cmp);
  if (mask != 0){
    return true;
  }
  return false;
}

template<typename T>
bool collides(const Sphere<T> &l, const std::array<Sphere<T>, 8> &rs){
  std::array<bool, 8> coll{false};

  //for (const auto &r: rs){
  //#pragma omp parallel for
  for (std::size_t i=0; i<8; ++i){
    const auto d = (l.x - rs[i].x) * (l.x - rs[i].x) + 
                   (l.y - rs[i].y) * (l.y - rs[i].y) + 
                   (l.z - rs[i].z) * (l.z - rs[i].z);

    coll[i] = d < (l.r + rs[i].r) * (l.r + rs[i].r);
  }

  if (std::any_of(coll.cbegin(), coll.cend(), [](const bool b){return b;})){
    return true;
  }
  return false;
}

template<typename T>
bool collides(const Sphere<T> &l, const SphereArray<T> &rs){
  std::array<bool, 8> coll{false};

  //for (const auto &r: rs){
  //#pragma omp parallel for
  for (std::size_t i=0; i<8; ++i){
    const T dx = (l.x - rs.x[i]);
    const T dy = (l.y - rs.y[i]);
    const T dz = (l.z - rs.z[i]);

    const T d = dx*dx + dy*dy + dz*dz;

    coll[i] = d < (l.r + rs.r) * (l.r + rs.r);
  }

  if (std::any_of(coll.cbegin(), coll.cend(), [](const bool b){return b;})){
    return true;
  }
  return false;
}

bool simd_collides(const Sphere<float> &l, const SphereArray<float> &rs){
  // x
  __m256 xl = _mm256_set1_ps(l.x);
  __m256 xr = _mm256_load_ps(rs.x.data());
  __m256 x = _mm256_sub_ps(xl, xr);
  __m256 x2 = _mm256_mul_ps(x, x);

  // y
  __m256 yl = _mm256_set1_ps(l.y);
  __m256 yr = _mm256_load_ps(rs.y.data());
  __m256 y = _mm256_sub_ps(yl, yr);
  __m256 y2 = _mm256_mul_ps(y, y);

  // z
  __m256 zl = _mm256_set1_ps(l.z);
  __m256 zr = _mm256_load_ps(rs.z.data());
  __m256 z = _mm256_sub_ps(zl, zr);
  __m256 z2 = _mm256_mul_ps(z, z);
  
  __m256 d = _mm256_add_ps(x2, y2);
  d = _mm256_add_ps(d, z2);

  /*{
    for (uint i=0; i<8; ++i){
      float a = (l.x - rs.x[i]);
      float b = (l.y - rs.y[i]);
      float c = (l.z - rs.z[i]);
      std::cout << a*a + b*b + c*c << " ";
    }
      std::cout << std::endl;
    float* res = (float*)&d;
    printf("%f %f %f %f %f %f %f %f\n",
      res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7]);
  }*/

  const float r = (l.r + rs.r);
  const float scalar_r2 = r * r;

  __m256 r2 = _mm256_set1_ps(scalar_r2);

  //std::cout << scalar_r2 << std::endl;

  //__m256 diff = _mm256_sub_ps(d, r2);

  //float* res = (float*)&diff;
  //printf("%f %f %f %f %f %f %f %f\n",
  //  res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7]);

  __m256 cmp = _mm256_cmp_ps(d, r2, _CMP_LT_OQ);

#if 1
  const int mask = _mm256_movemask_ps(cmp);
  if (mask != 0){
    return true;
  }
  return false;
#else

  float* res2 = (float*)&cmp;
  //printf("%d %d %d %d %d %d %d %d\n",
  //  res2[0], res2[1], res2[2], res2[3], res2[4], res2[5], res2[6], res2[7]);

  for (uint i=0; i<8; ++i){
    if (res2[i] == 0xffffffff){
      return true;
    }
  }

  return false;
#endif
}

template <typename T>
using Edge = std::pair<Configuration<T>, Configuration<T>>;

template<typename T>
struct Environment{
  std::vector<Sphere<T>> sphere_obstacles;
  std::vector<AABB<T>> aabb_obstacles;

  bool is_valid(const Configuration<T> &q) const{
    // update robot position
    Sphere<T> robot;
    robot.r = 0.5;
    robot.x = q.x;
    robot.y = q.y;
    robot.z = q.z;

    bool valid = true;
    // collide all objects
    for (const auto &o: sphere_obstacles){
      valid = valid && !collides(o, robot);

      if (!valid){
        return false;
      }
    }

    for (const auto &o: aabb_obstacles){
      valid = valid && !collides(o, robot);

      if (!valid){
        return false;
      }
    }

    return valid;
  }

  bool is_valid(const std::array<Configuration<T>, 8> &qs) const{
    std::array<Sphere<T>, 8> rs;
    for (std::size_t i=0; i<8; ++i){
      rs[i].x = qs[i].x;
      rs[i].y = qs[i].y;
      rs[i].z = qs[i].z;
      rs[i].r = 0.5;
    }

    bool valid = true;
    for (const auto &obs: sphere_obstacles){
      valid = valid && !collides(obs, rs);
      if (!valid){
        return false;
      }
    }
    for (const auto &obs: aabb_obstacles){
      //valid = valid && !collides(obs, rs);
      std::cout << "not implemented" << std::endl;
      if (!valid){
        return false;
      }
    }

    return valid;
  }

  bool is_valid(const ConfigurationOfArrays<T> &qs) const{
    SphereArray<T> rs;
    rs.x = qs.x;
    rs.y = qs.y;
    rs.z = qs.z;
    rs.r = 0.5;

    bool valid = true;
    for (const auto &obs: sphere_obstacles){
      valid = valid && !collides(obs, rs);
      if (!valid){
        return false;
      }
    }
    for (const auto &obs: aabb_obstacles){
      //valid = valid && !collides(obs, rs);
      std::cout << "not implemented" << std::endl;
      if (!valid){
        return false;
      }
    }
    return valid;
  }

  bool is_valid_simd(const ConfigurationOfArrays<float> &qs) const{
    SphereArray<float> rs;
    rs.x = qs.x;
    rs.y = qs.y;
    rs.z = qs.z;
    rs.r = 0.5;

    bool valid = true;
    for (const auto &obs: sphere_obstacles){
      valid = valid && !simd_collides(obs, rs);
      if (!valid){
        return false;
      }
    }
    for (const auto &obs: aabb_obstacles){
      valid = valid && !simd_collides(obs, rs);
      if (!valid){
        return false;
      }
    }
    return valid;
  }

  // in some settings, doing many additions, instead of comuting the value directly
  // leads to slight differences in rounding, leading to different locations that are collision
  // checked
  bool check_edge_linear(const Edge<T> &edge, const uint num_pts) const{
    // deltas
    const float xd = (edge.second.x - edge.first.x) / (num_pts + 1);
    const float yd = (edge.second.y - edge.first.y) / (num_pts + 1);
    const float zd = (edge.second.z - edge.first.z) / (num_pts + 1);
    
    // initialize
#if 1
    Configuration<T> q;
    q.x = edge.first.x + xd;
    q.y = edge.first.y + yd;
    q.z = edge.first.z + zd;
    
    for (uint i=0; i<num_pts; ++i){
      q.x += xd;
      q.y += yd;
      q.z += zd;

      const bool valid = is_valid(q);
      if (!valid){
        return false;
      }
    }
    return true;
#else
    // initialize
    Configuration<T> q;
    
    for (uint i=0; i<num_pts; ++i){
      q.x = edge.first.x + xd * (i+1);
      q.y = edge.first.y + yd * (i+1);
      q.z = edge.first.z + zd * (i+1);

      const bool valid = is_valid(q);
      if (!valid){
        return false;
      }
    }
    return true;
#endif
  }

  bool check_edge_rake(const Edge<T> &edge, const uint num_pts) const{
    const float xd = (edge.second.x - edge.first.x) / (num_pts + 1);
    const float yd = (edge.second.y - edge.first.y) / (num_pts + 1);
    const float zd = (edge.second.z - edge.first.z) / (num_pts + 1);

    const int rake_length = num_pts/8;
    Configuration<T> q;
    for (uint i=0; i<rake_length; ++i){
      for (uint j=0; j<8; ++j){
        const uint idx = i + j*rake_length;
        q.x = edge.first.x + (idx+1) * xd;
        q.y = edge.first.y + (idx+1) * yd;
        q.z = edge.first.z + (idx+1) * zd;

        const bool valid = is_valid(q);
        if (!valid){
          return false;
        }
      }
    }
    return true;
  }

  bool check_edge_batched(const Edge<T> &edge, const uint num_pts) const{
    std::array<Configuration<T>, 8> qs;

    for (uint i=0; i<num_pts; i+=8){
      for (uint j=0; j<8; ++j){
        const uint idx = i + j;
        qs[j].x = edge.first.x + (idx+1) * (edge.second.x - edge.first.x) / (num_pts + 1);
        qs[j].y = edge.first.y + (idx+1) * (edge.second.y - edge.first.y) / (num_pts + 1);
        qs[j].z = edge.first.z + (idx+1) * (edge.second.z - edge.first.z) / (num_pts + 1);
      }
      const bool valid = is_valid(qs);
      if (!valid){
        return false;
      }
    }

    return true;
  }

  bool check_edge_batched_memlayout(const Edge<T> &edge, const uint num_pts) const{
    ConfigurationOfArrays<float> qs;
    return false;
  }

  bool check_edge_simd(const Edge<float> &edge, const uint num_pts) const{
    const float xd = (edge.second.x - edge.first.x) / (num_pts + 1);
    const float yd = (edge.second.y - edge.first.y) / (num_pts + 1);
    const float zd = (edge.second.z - edge.first.z) / (num_pts + 1);

    ConfigurationOfArrays<float> qs;
    for (uint i=0; i<num_pts; i+=8){
      // this can be parallelized
      for (uint j=0; j<8; ++j){
        const uint idx = i + j;
        qs.x[j] = edge.first.x + (idx+1) * xd;
        qs.y[j] = edge.first.y + (idx+1) * yd;
        qs.z[j] = edge.first.z + (idx+1) * zd;
      }

      const bool valid = is_valid_simd(qs);
      if (!valid){
        return false;
      }
    }
    return true;
  }

  bool check_edge_simd_rake(const Edge<float> &edge, const uint num_pts) const{
    const float xd = (edge.second.x - edge.first.x) / (num_pts + 1);
    const float yd = (edge.second.y - edge.first.y) / (num_pts + 1);
    const float zd = (edge.second.z - edge.first.z) / (num_pts + 1);

    // initialize
    const int rake_length = num_pts/8;
    ConfigurationOfArrays<float> qs;

    for (uint j=0; j<8; ++j){
      qs.x[j] = edge.first.x + xd * (1 + rake_length * j);
      qs.y[j] = edge.first.y + yd * (1 + rake_length * j);
      qs.z[j] = edge.first.z + zd * (1 + rake_length * j);
    }

    for (uint i=0; i<rake_length; ++i){
      // this can be parallelized, and it might avoid reloading registers
      for (uint j=0; j<8; ++j){
        qs.x[j] += xd;
        qs.y[j] += yd;
        qs.z[j] += zd;
      }

      const bool valid = is_valid_simd(qs);
      if (!valid){
        return false;
      }
    }
    return true;
  }
};

template<typename T>
void sequential(Environment<T> env, const std::vector<Configuration<T>> &qs){
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  //#pragma omp parallel for
  for (std::size_t i=0; i<qs.size(); ++i){
    res = env.is_valid(qs[i]);
  }

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time difference (sec) = " << 
    (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0  <<std::endl;
}

template<typename T>
void sequential_edges(Environment<T> env, const std::vector<Edge<T>> &edges, const uint num_pts){
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  //#pragma omp parallel for
  for (std::size_t i=0; i<edges.size(); ++i){
    res = env.check_edge_linear(edges[i], num_pts);
  }

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time difference (sec) = " << 
    (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0  <<std::endl;
}

template<typename T>
void sequential_edges_rake(Environment<T> env, const std::vector<Edge<T>> &edges, const uint num_pts){
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  //#pragma omp parallel for
  for (std::size_t i=0; i<edges.size(); ++i){
    res = env.check_edge_rake(edges[i], num_pts);
  }

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time difference (sec) = " << 
    (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0  <<std::endl;
}

template<typename T>
void v2(Environment<T> env, const std::vector<Configuration<T>> &qs){
  std::array<Configuration<T>, 8> buffer;
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  for (std::size_t i=0; i<qs.size(); i+=8){
    for (std::size_t j=0; j<8; ++j){
      buffer[j] = qs[i+j];
    }
    res = env.is_valid(buffer);
  }

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time difference (sec) = " << 
    (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0  <<std::endl;
}

template<typename T>
void batched_edges(Environment<T> env, const std::vector<Edge<T>> &edges, const uint num_pts){
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  for (std::size_t i=0; i<edges.size(); ++i){
    res = env.check_edge_batched(edges[i], num_pts);
  }

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time difference (sec) = " << 
    (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0  <<std::endl;
}

template<typename T>
void v3(Environment<T> env, const std::vector<Configuration<T>> &qs){
  ConfigurationOfArrays<T> buffer;
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  for (std::size_t i=0; i<qs.size(); i+=8){
    for (std::size_t j=0; j<8; ++j){
      buffer.x[j] = qs[i+j].x;
      buffer.y[j] = qs[i+j].y;
      buffer.z[j] = qs[i+j].z;
    }
    res = env.is_valid(buffer);
  }

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time difference (sec) = " << 
    (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0  <<std::endl;
}

void simd(Environment<float> env, const std::vector<Configuration<float>> &qs){
  ConfigurationOfArrays<float> buffer;
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  for (std::size_t i=0; i<qs.size(); i+=8){
    for (std::size_t j=0; j<8; ++j){
      buffer.x[j] = qs[i+j].x;
      buffer.y[j] = qs[i+j].y;
      buffer.z[j] = qs[i+j].z;
    }
    res = env.is_valid_simd(buffer);
  }

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time difference (sec) = " << 
    (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0  <<std::endl;
}

void simd_edges(Environment<float> env, const std::vector<Edge<float>> &edges, const uint num_pts){
  ConfigurationOfArrays<float> buffer;
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  for (std::size_t i=0; i<edges.size(); ++i){
    res = env.check_edge_simd(edges[i], num_pts);
  }

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time difference (sec) = " << 
    (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0  <<std::endl;
}

void simd_edges_rake(Environment<float> env, const std::vector<Edge<float>> &edges, const uint num_pts){
  ConfigurationOfArrays<float> buffer;
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
  for (std::size_t i=0; i<edges.size(); ++i){
    res = env.check_edge_simd_rake(edges[i], num_pts);
  }

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Time difference (sec) = " << 
    (std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()) /1000000.0  <<std::endl;
}

template<typename T>
Environment<T> make_sphere_environment(const uint n, const uint seed=0){
  Environment<T> env;

  std::mt19937 gen(seed); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis(-5., 5.);
  std::uniform_real_distribution<> size(0, 0.5);

  for (std::size_t i=0; i<0; ++i){
    Sphere<T> obs;
    obs.x = dis(gen);
    obs.y = dis(gen);
    obs.z = dis(gen);

    obs.r = 0.1;

    env.sphere_obstacles.push_back(obs);
  }

  for (std::size_t i=0; i<100; ++i){
    AABB<T> obs;
    obs.xmin = dis(gen);
    obs.ymin = dis(gen);
    obs.zmin = dis(gen);

    obs.xmax = obs.xmin + size(gen);
    obs.ymax = obs.ymin + size(gen);
    obs.zmax = obs.zmin + size(gen);

    env.aabb_obstacles.push_back(obs);
  }

  return env;
}

template<typename T>
std::vector<Configuration<T>> make_random_configurations(const uint n, const uint seed=0){
  std::mt19937 gen(seed); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis(-5., 5.);

  std::vector<Configuration<T>> qs;

  for (std::size_t i=0; i<n; ++i){
    // random position
    Configuration<T> q;
    q.x = dis(gen);
    q.y = dis(gen);
    q.z = dis(gen);
    
    qs.push_back(q);
  }

  return qs;
}

template<typename T>
std::vector<Edge<T>> make_edges(const uint num_edges, const uint seed=0){
  std::mt19937 gen(seed); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis(-5., 5.);

  std::vector<Edge<T>> edges;

  for (uint i=0; i<num_edges; ++i){
    std::vector<Configuration<T>> qs;
    // sample start and endpoint - currently simply random, not 
    Configuration<T> q0;
    q0.x = dis(gen);
    q0.y = dis(gen);
    q0.z = dis(gen);

    Configuration<T> q1;
    q1.x = dis(gen);
    q1.y = dis(gen);
    q1.z = dis(gen);

    Edge<T> e;
    e.first = q0;
    e.second = q1;

    edges.push_back(e);
  }

  return edges;
}

void test_configurations(){
  const uint num_qs = 8000;
  Environment<float> env = make_sphere_environment<float>(10);
  std::vector<Configuration<float>> qs = make_random_configurations<float>(num_qs);

  std::vector<bool> v1_results;
  for (std::size_t i=0; i<qs.size(); i+=8){
    bool tmp_res = true;
    for (std::size_t j=0; j<8; ++j){
      tmp_res = tmp_res && env.is_valid(qs[i+j]);
    }
    v1_results.push_back(tmp_res);
  }

  std::vector<bool> v3_results;
  {
    ConfigurationOfArrays<float> buffer;
    for (std::size_t i=0; i<qs.size(); i+=8){
      for (std::size_t j=0; j<8; ++j){
        buffer.x[j] = qs[i+j].x;
        buffer.y[j] = qs[i+j].y;
        buffer.z[j] = qs[i+j].z;
      }
      v3_results.push_back(env.is_valid(buffer));
    }
  }

  std::vector<bool> v4_results;
  {
    ConfigurationOfArrays<float> buffer;
    for (std::size_t i=0; i<qs.size(); i+=8){
      for (std::size_t j=0; j<8; ++j){
        buffer.x[j] = qs[i+j].x;
        buffer.y[j] = qs[i+j].y;
        buffer.z[j] = qs[i+j].z;
      }
      v4_results.push_back(env.is_valid_simd(buffer));
    }
  }

  for (uint i=0; i<v4_results.size(); ++i){
    //std::cout << v1_results[i] << std::endl;

    if(v1_results[i] != v3_results[i]){
      std::cout << v1_results[i] << " " <<  v3_results[i] << std::endl;
      std::cout << "v1 vs v3" << std::endl; 
    }

    if(v1_results[i] != v4_results[i]){
      std::cout << v1_results[i] << " " <<  v4_results[i] << std::endl;
      std::cout << "v1 vs v4" << std::endl; 
    }
  }
}

void test_edges(){
  const uint num_obstacles = 50;
  Environment<float> env = make_sphere_environment<float>(num_obstacles);

  const uint num_edges = 10000;
  std::vector<Edge<float>> edges = make_edges<float>(num_edges);

  const uint num_pts = 160;
  
  std::vector<bool> sequential_linear_results;
  for (const auto &edge: edges){
    sequential_linear_results.push_back(env.check_edge_linear(edge, num_pts));
  }

  std::vector<bool> sequential_rake_results;
  for (const auto &edge: edges){
    sequential_rake_results.push_back(env.check_edge_rake(edge, num_pts));
  }

  std::vector<bool> simd_results;
  for (const auto &edge: edges){
    simd_results.push_back(env.check_edge_simd(edge, num_pts));
  }

  std::vector<bool> simd_rake_results;
  for (const auto &edge: edges){
    simd_rake_results.push_back(env.check_edge_simd_rake(edge, num_pts));
  }

  for (uint i=0; i<num_edges; ++i){
    if (sequential_linear_results[i] != sequential_rake_results[i]){
      std::cout << "seq " << sequential_linear_results[i] << " rake " << sequential_rake_results[i] << std::endl;
    }

    if (sequential_linear_results[i] != simd_results[i]){
      std::cout << "seq " << sequential_linear_results[i] << " simd " << simd_results[i] << std::endl;
    }

    if (sequential_linear_results[i] != simd_rake_results[i]){
      std::cout << "seq simd rake" << std::endl;
    }
  }
}

void benchmark_random(){
  const uint num_obstacles = 100;
  Environment<float> float_env = make_sphere_environment<float>(num_obstacles);
  std::vector<Configuration<float>> float_qs = make_random_configurations<float>(8000000);

  Environment<double> double_env = make_sphere_environment<double>(num_obstacles);
  std::vector<Configuration<double>> double_qs = make_random_configurations<double>(8000000);

  // warmup
  sequential<float>(float_env, float_qs);
  sequential<double>(double_env, double_qs);

  std::cout << "sequential" << std::endl;
  sequential<float>(float_env, float_qs);
  sequential<double>(double_env, double_qs);

  std::cout << "batched, not parallel" << std::endl;
  //v2<float>(float_env, float_qs);
  //v2<double>(double_env, double_qs);

  std::cout << "batched, other memlayout" << std::endl;
  //v3<float>(float_env, float_qs);
  //v3<double>(double_env, double_qs);

  std::cout << "batched, simd" << std::endl;
  simd(float_env, float_qs);

  std::cout << res << std::endl;
}

void benchmark_edges(){
  const uint num_obstacles = 100;
  Environment<float> env = make_sphere_environment<float>(num_obstacles);

  const uint num_edges = 100000;
  std::vector<Edge<float>> edges = make_edges<float>(num_edges);

  const uint num_pts = 160;
  
  std::cout << "sequential (linear)" << std::endl;
  sequential_edges<float>(env, edges, num_pts);

  std::cout << "sequential (rake)" << std::endl;
  sequential_edges<float>(env, edges, num_pts);

  std::cout << "batched (linear)" << std::endl;
  //batched_edges(env, edges, num_pts);

  std::cout << "simd (linear)" << std::endl;
  simd_edges(env, edges, num_pts);

  std::cout << "simd (rake)" << std::endl;
  simd_edges_rake(env, edges, num_pts);

  std::cout << res << std::endl;
}

int main(){
  //test_configurations();
  //test_edges();

  benchmark_random();
  benchmark_edges();
}
