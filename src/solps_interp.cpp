#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <iterator>
#include <string>
#include <tuple>
#include<stdio.h>
#include <netcdf>
#include <vector>
#include <cmath>
#include <chrono>
#include <omp.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#ifdef __CUDACC__
#define DEVICE_CALLABLE __host__ __device__
#else
#define DEVICE_CALLABLE
#endif

class Managed {
public:
 void *operator new(size_t len) {
 void *ptr;
 cudaMallocManaged(&ptr, len);
 cudaDeviceSynchronize();
 return ptr;
 }
 void operator delete(void *ptr) {
 //cudaDeviceSynchronize();
 cudaFree(ptr);
 }
};

namespace sim {
template<typename T>
  class Vector : public Managed {

    // @todo constexpr in c++14
    T *alloc_data() {
#if defined(CUDA)
      T* data;
      auto err = cudaMallocManaged(&data, sizeof(T)*capacity_);
      if(err != cudaSuccess){
        throw std::runtime_error("error allocating managed memory");
      }
      return data;
#else
      return new T[capacity_];
#endif
    }

    // @todo constexpr in c++14
    void free_data() {
#if defined(CUDA)
      cudaFree((void*)data_);
#else
      delete[] data_;
#endif
    }

  public:
    /*! Construct an vector of fixed capacity
     */
    Vector() : capacity_{1}, size_{1}, data_{alloc_data()} {}

    Vector(const std::size_t capacity) : capacity_{capacity}, size_{capacity}, data_{alloc_data()} {}

    /*! Construct an Vector of fixed capacity and initialize values
     */
    Vector(const std::size_t capacity, T initial_value) : capacity_{capacity}, size_{capacity}, data_{alloc_data()} {
        for (std::size_t i = 0; i < size_; i++) {
        data_[i] = initial_value;
      }
    }

    Vector(const std::vector<T> initial_vector) : capacity_{initial_vector.size()}, size_{initial_vector.size()}, data_{alloc_data()} {
        for (std::size_t i = 0; i < size_; i++) {
        data_[i] = initial_vector[i];
      }
    }


    /*! Destruct vector memory
     */
    ~Vector() {
      free_data();
    }

    /*! Copy constructor
     */
    Vector(const Vector &source) {
      capacity_ = source.capacity_;
      size_ = source.size_;
      data_ = alloc_data();

      if (data_) {
        for (std::size_t i = 0; i < size_; i++) {
          data_[i] = source[i];
        }
      }
    }

    Vector &operator=(const Vector &source)// = delete;
    {
        for(int i=0;i<source.size();i++)
        {
            data_[i] = source[i];
        }
        return *this;
    }

    Vector &operator=(const std::vector<T> &source)// = delete;
    {
        for(int i=0;i<source.size();i++)
        {
            data_[i] = source[i];
        }
        return *this;
    }
    
    Vector &operator=(const T source)// = delete;
    {
        for(int i=0;i<capacity_;i++)
        {
            data_[i] = source;
        }
        return *this;
    }
    Vector(Vector &&) noexcept = delete;

    Vector &operator=(Vector &&)      = delete;

    /*! Vector size getter
     * @return the number of in use elements in the Vector
     */
    DEVICE_CALLABLE
    std::size_t size() const {
      return size_;
    }

    /*! Vector capacity getter
     * @return the maximum number of elements in the Vector
     */
    std::size_t capacity() const {
      return capacity_;
    }

    /*! Vector capacity getter
     * @return the maximum number of elements in the Vector
     */
    std::size_t available() const {
      return capacity() - size();
    }

    /*! Return reference to first element of Vector
     * @return reference to first element
     */
    T &front() {
      return data_[0];
    }
    /*
    *  This function will %resize the %vector to the specified
               *         *  number of elements.  If the number is smaller than the
               *                *  %vector's current size the %vector is truncated, otherwise
               *                       *  default constructed elements are appended.
               *                              */
    void resize(const T __new_size)
    {
      if (__new_size > size())
      {
          free_data();
          capacity_ = __new_size;
          size_ = __new_size;
          data_ = alloc_data();

      }
      else if (__new_size < size())
      {
       
      }
    }

    /*! Getter for pointer to underlying data
     */
    DEVICE_CALLABLE
    T *data() {
      return this->data_;
    }

    /*! const getter for pointer to underlying data
     */
    DEVICE_CALLABLE
    T *data() const {
      return this->data_;
    }

    /*! Subscript operator, []
     * Retrieve reference to element using subscript notation
     */
    DEVICE_CALLABLE
    T &operator[](const std::size_t index) {
      return data_[index];
    }

    /*! const subscript operator, []
     *  Retrieve const reference to element using subscript notation
     */
    DEVICE_CALLABLE
    const T &operator[](const std::size_t index) const {
      return data_[index];
    }
/*
    DEVICE_CALLABLE
    const T* begin() {
      return this->data();
    }
    DEVICE_CALLABLE
    const T* end() {
      return this->data() + this->size();
    }
*/
    DEVICE_CALLABLE
    T *begin() const {
      return this->data();
    }

    DEVICE_CALLABLE
    T *end() const {
      return this->data() + this->size();
    }


    private:
// @todo DEVICE_CALLABLE cant use private member variables
//  public:
    std::size_t capacity_;
    std::size_t size_;
    T *data_;

  };

  /*! begin iterator for range based for loops
   */
  template<typename T>
  DEVICE_CALLABLE
  const T *begin(const Vector<T> &vector) {
    return vector.data();
  }

  /*! end iterator for range based for loops
   */
  template<typename T>
  DEVICE_CALLABLE
  const T *end(const Vector<T> &vector) {
    return vector.data() + vector.size();
  }
}

struct Fields : public Managed
{
	sim::Vector<double> te1t;
	sim::Vector<double> te2t;
	sim::Vector<double> te3t;
	sim::Vector<double> te;
	
	sim::Vector<double> ti1t;
	sim::Vector<double> ti2t;
	sim::Vector<double> ti3t;
	sim::Vector<double> ti;
	
	sim::Vector<double> ni1t;
	sim::Vector<double> ni2t;
	sim::Vector<double> ni3t;
	sim::Vector<double> ni;
	
	sim::Vector<double> ne1t;
	sim::Vector<double> ne2t;
	sim::Vector<double> ne3t;
	sim::Vector<double> ne;
	//sim::Vector<double> flux_last1t;
	//sim::Vector<double> flux_last2t;
	//sim::Vector<double> flux_last3t;
	//sim::Vector<double> flux_last;
	
	sim::Vector<double> mass1t;
	sim::Vector<double> mass2t;
	sim::Vector<double> mass3t;
	sim::Vector<double> mass;
	
	sim::Vector<double> charge1t;
	sim::Vector<double> charge2t;
	sim::Vector<double> charge3t;
	sim::Vector<double> charge;
	
	sim::Vector<double> Br1t;
	sim::Vector<double> Br2t;
	sim::Vector<double> Br3t;
	sim::Vector<double> Br;
	
	sim::Vector<double> Bt1t;
	sim::Vector<double> Bt2t;
	sim::Vector<double> Bt3t;
	sim::Vector<double> Bt;
	
	sim::Vector<double> Bz1t;
	sim::Vector<double> Bz2t;
	sim::Vector<double> Bz3t;
	sim::Vector<double> Bz;
	
	sim::Vector<double> Bmag1t;
	sim::Vector<double> Bmag2t;
	sim::Vector<double> Bmag3t;
	sim::Vector<double> Bmag;
	
	sim::Vector<double> vr1t;
	sim::Vector<double> vr2t;
	sim::Vector<double> vr3t;
	sim::Vector<double> vr;
	
	sim::Vector<double> vt1t;
	sim::Vector<double> vt2t;
	sim::Vector<double> vt3t;
	sim::Vector<double> vt;
	
	sim::Vector<double> vz1t;
	sim::Vector<double> vz2t;
	sim::Vector<double> vz3t;
	sim::Vector<double> vz;
	
	sim::Vector<double> Er1t;
	sim::Vector<double> Er2t;
	sim::Vector<double> Er3t;
	sim::Vector<double> Er;
	
	sim::Vector<double> Ez1t;
	sim::Vector<double> Ez2t;
	sim::Vector<double> Ez3t;
	sim::Vector<double> Ez;

	sim::Vector<double> gradTe1t;
	sim::Vector<double> gradTe2t;
	sim::Vector<double> gradTe3t;
	sim::Vector<double> gradTe;
	
	sim::Vector<double> gradTer1t;
	sim::Vector<double> gradTer2t;
	sim::Vector<double> gradTer3t;
	sim::Vector<double> gradTer;
	
	sim::Vector<double> gradTet1t;
	sim::Vector<double> gradTet2t;
	sim::Vector<double> gradTet3t;
	sim::Vector<double> gradTet;
	
	sim::Vector<double> gradTez1t;
	sim::Vector<double> gradTez2t;
	sim::Vector<double> gradTez3t;
	sim::Vector<double> gradTez;
	
	sim::Vector<double> gradTi1t;
	sim::Vector<double> gradTi2t;
	sim::Vector<double> gradTi3t;
	sim::Vector<double> gradTi;
	
	sim::Vector<double> gradTir1t;
	sim::Vector<double> gradTir2t;
	sim::Vector<double> gradTir3t;
	sim::Vector<double> gradTir;
	
	sim::Vector<double> gradTit1t;
	sim::Vector<double> gradTit2t;
	sim::Vector<double> gradTit3t;
	sim::Vector<double> gradTit;
	
	sim::Vector<double> gradTiz1t;
	sim::Vector<double> gradTiz2t;
	sim::Vector<double> gradTiz3t;
	sim::Vector<double> gradTiz;
	//CUDA_CALLABLE_MEMBER
  	//Fields() :
        //nParticles{getVariable_cfg<unsigned int> (cfg,"impurityParticleSource.nP")},
	
};

void vectorPrint(std::string a, double D[])
{
    std::cout << "vector " << a << " " << 
    D[0] << " " <<  
    D[1] << " " << 
    D[2] << std::endl;
}

__host__ __device__
void vectorAssign(double a, double b,double c, double D[])
{
    D[0] = a;
    D[1] = b;
    D[2] = c;
}

__host__ __device__
void vectorSubtract(double A[], double B[],double C[])
{
    C[0] = A[0] - B[0];
    C[1] = A[1] - B[1];
    C[2] = A[2] - B[2];
}

__host__ __device__
void vectorCrossProduct(double A[], double B[], double C[])
{
    double tmp[3] = {0.0f,0.0f,0.0f};
    tmp[0] = A[1]*B[2] - A[2]*B[1];
    tmp[1] = A[2]*B[0] - A[0]*B[2];
    tmp[2] = A[0]*B[1] - A[1]*B[0];

    C[0] = tmp[0];
    C[1] = tmp[1];
    C[2] = tmp[2];
}

__host__ __device__
double vectorDotProduct(double A[], double B[])
{
    double c = A[0]*B[0] +  A[1]*B[1] + A[2]*B[2];
    return c;
}

__host__ __device__
double vectorNorm(double A[])
{
    double norm = 0.0f;
    norm = std::sqrt(A[0]*A[0] + A[1]*A[1] + A[2]*A[2]);

        return norm;
}

__host__ __device__
double interpolate_value(double r1, double z1, double a1, double r2, double z2, double a2, double r3, double z3, double a3, double r_point, double z_point)
{
                    double denom = (z2 - z3)*(r1 - r3) + (r3 - r2)*(z1 - z3);
                    double weights0 = ((z2 - z3)*(r_point - r3) + (r3 - r2)*(z_point - z3))/denom;
                    double weights1 = ((z3 - z1)*(r_point - r3) + (r1 - r3)*(z_point - z3))/denom;
                    double weights2 = 1.0 - weights0 - weights1;
                    double a_point = a1*weights0 + a2*weights1 + a3*weights2;

                    return a_point;
}

__host__ __device__
bool in_triangle(double r1, double z1, double r2, double z2, double r3, double z3, double r_point, double z_point)
{
      bool in_triangle = false;
      double A[3] = {0.0, 0.0, 0.0};
      double B[3] = {0.0, 0.0, 0.0};
      double C[3] = {0.0, 0.0, 0.0};
      double AB[3] = {0.0, 0.0, 0.0};
      double AC[3] = {0.0, 0.0, 0.0};
      double BC[3] = {0.0, 0.0, 0.0};
      double CA[3] = {0.0, 0.0, 0.0};
      double p[3] = {r_point, 0.0, z_point};
      double Ap[3] = {0.0, 0.0, 0.0};
      double Bp[3] = {0.0, 0.0, 0.0};
      double Cp[3] = {0.0, 0.0, 0.0};
      double normalVector[3] = {0.0, 0.0, 0.0};
      double crossABAp[3] = {0.0, 0.0, 0.0};
      double crossBCBp[3] = {0.0, 0.0, 0.0};
      double crossCACp[3] = {0.0, 0.0, 0.0};
      double signDot0 = 0.0;
      double signDot1 = 0.0;
      double signDot2 = 0.0;
      double totalSigns = 0.0;
      
          vectorAssign(r1, 0.0,z1, A);
          vectorAssign(r2, 0.0,z2, B);
          vectorAssign(r3, 0.0,z3, C);
          //vectorPrint("A " , A);
          //vectorPrint("B " , B);
          //vectorPrint("C " , C);
          vectorSubtract(B, A, AB);
          vectorSubtract(C, A, AC);
          vectorSubtract(C, B, BC);
          vectorSubtract(A, C, CA);

          vectorSubtract(p, A, Ap);
          vectorSubtract(p, B, Bp);
          vectorSubtract(p, C, Cp);

          vectorCrossProduct(AB, AC, normalVector);
          vectorCrossProduct(AB, Ap, crossABAp);
          vectorCrossProduct(BC, Bp, crossBCBp);
          vectorCrossProduct(CA, Cp, crossCACp);
          
          signDot0 =
              std::copysign(1.0, vectorDotProduct(crossABAp, normalVector));
          signDot1 =
              std::copysign(1.0, vectorDotProduct(crossBCBp, normalVector));
          signDot2 =
              std::copysign(1.0, vectorDotProduct(crossCACp, normalVector));
          totalSigns = 1.0 * std::abs(signDot0 + signDot1 + signDot2);
          if (totalSigns == 3.0) {
            in_triangle = true;
          }
          if (vectorNorm(crossABAp) == 0.0 || vectorNorm(crossBCBp) == 0.0 ||
              vectorNorm(crossCACp) == 0.0) {
            in_triangle = true;
          }

          return in_triangle;
}

struct saxpy_functor
{
    int* index_triangle;
    int nr;
    int nz;
    double* r;
    double* z;
    double* r1;
    double* z1;
    double* v1;
    double* r2;
    double* z2;
    double* v2;
    double* r3;
    double* z3;
    double* v3;
    double* radius;
    double* val;
    bool* found;
    Fields* solps_fields;

    saxpy_functor(int* _index_triangle, int _nr, int _nz,double* _r, double* _z,
        double* _r1, double* _z1, double* _v1,
        double* _r2, double* _z2, double* _v2,
        double* _r3, double* _z3, double* _v3,
        double* _radius, double* _val, bool* _found, Fields* _solps_fields) : 
      index_triangle(_index_triangle), nr(_nr), nz(_nz), r(_r), z(_z),
      r1(_r1), z1(_z1), v1(_v1),
      r2(_r2), z2(_z2), v2(_v2),
      r3(_r3), z3(_z3), v3(_v3),
      radius (_radius), val(_val), found(_found), solps_fields(_solps_fields) {}

    __host__ __device__
        void operator()(size_t index) const {
            int jj = floor(index/nr);
            int ii = index%nr;
            int i = index_triangle[0];
            double dist = (r[ii] - r1[i])*(r[ii] - r1[i]) + (z[jj] - z1[i])*(z[jj] - z1[i]);
            
            if(dist < radius[i]*radius[i])
            {
               bool in = in_triangle(r1[i],z1[i],r2[i],z2[i],r3[i],z3[i],r[ii],z[jj]);
	       if(in)
               {
                 double a_point = interpolate_value(r1[i],z1[i],v1[i],
                                                    r2[i],z2[i],v2[i],
                                                    r3[i],z3[i],v3[i],
                                                    r[ii],z[jj]);
                 val[index] = a_point;
                 found[index] = false;
                 a_point = interpolate_value(r1[i],z1[i],solps_fields->te1t[i],
                                             r2[i],z2[i],solps_fields->te2t[i],
                                             r3[i],z3[i],solps_fields->te3t[i],
                                             r[ii],z[jj]);
		 solps_fields->te[index] = a_point;
		 
		 a_point = interpolate_value(r1[i],z1[i],solps_fields->ti1t[i],
                                             r2[i],z2[i],solps_fields->ti2t[i],
                                             r3[i],z3[i],solps_fields->ti3t[i],
                                             r[ii],z[jj]);
		 solps_fields->ti[index] = a_point;
                 
		 a_point = interpolate_value(r1[i],z1[i],solps_fields->ni1t[i],
                                             r2[i],z2[i],solps_fields->ni2t[i],
                                             r3[i],z3[i],solps_fields->ni3t[i],
                                             r[ii],z[jj]);
		 solps_fields->ni[index] = a_point;
		 
		 a_point = interpolate_value(r1[i],z1[i],solps_fields->ne1t[i],
                                             r2[i],z2[i],solps_fields->ne2t[i],
                                             r3[i],z3[i],solps_fields->ne3t[i],
                                             r[ii],z[jj]);
		 solps_fields->ne[index] = a_point;
		 
		 //a_point = interpolate_value(r1[i],z1[i],solps_fields->flux_last1t[i],
                 //                            r2[i],z2[i],solps_fields->flux_last2t[i],
                 //                            r3[i],z3[i],solps_fields->flux_last3t[i],
                 //                            r[ii],z[jj]);
		 //solps_fields->flux_last[index] = a_point;
		 
		 a_point = interpolate_value(r1[i],z1[i],solps_fields->mass1t[i],
                                             r2[i],z2[i],solps_fields->mass2t[i],
                                             r3[i],z3[i],solps_fields->mass3t[i],
                                             r[ii],z[jj]);
		 solps_fields->mass[index] = a_point;
		 
		 a_point = interpolate_value(r1[i],z1[i],solps_fields->charge1t[i],
                                             r2[i],z2[i],solps_fields->charge2t[i],
                                             r3[i],z3[i],solps_fields->charge3t[i],
                                             r[ii],z[jj]);
		 solps_fields->charge[index] = a_point;
		 
		 a_point = interpolate_value(r1[i],z1[i],solps_fields->Br1t[i],
                                             r2[i],z2[i],solps_fields->Br2t[i],
                                             r3[i],z3[i],solps_fields->Br3t[i],
                                             r[ii],z[jj]);
		 solps_fields->Br[index] = a_point;
		 
		 a_point = interpolate_value(r1[i],z1[i],solps_fields->Bt1t[i],
                                             r2[i],z2[i],solps_fields->Bt2t[i],
                                             r3[i],z3[i],solps_fields->Bt3t[i],
                                             r[ii],z[jj]);
		 solps_fields->Bt[index] = a_point;
		 
		 a_point = interpolate_value(r1[i],z1[i],solps_fields->Bz1t[i],
                                             r2[i],z2[i],solps_fields->Bz2t[i],
                                             r3[i],z3[i],solps_fields->Bz3t[i],
                                             r[ii],z[jj]);
		 solps_fields->Bz[index] = a_point;
		 
		 a_point = interpolate_value(r1[i],z1[i],solps_fields->Bmag1t[i],
                                             r2[i],z2[i],solps_fields->Bmag2t[i],
                                             r3[i],z3[i],solps_fields->Bmag3t[i],
                                             r[ii],z[jj]);
		 solps_fields->Bmag[index] = a_point;
		 
		 a_point = interpolate_value(r1[i],z1[i],solps_fields->vr1t[i],
                                             r2[i],z2[i],solps_fields->vr2t[i],
                                             r3[i],z3[i],solps_fields->vr3t[i],
                                             r[ii],z[jj]);
		 solps_fields->vr[index] = a_point;
		 
		 a_point = interpolate_value(r1[i],z1[i],solps_fields->vt1t[i],
                                             r2[i],z2[i],solps_fields->vt2t[i],
                                             r3[i],z3[i],solps_fields->vt3t[i],
                                             r[ii],z[jj]);
		 solps_fields->vt[index] = a_point;
		 
		 a_point = interpolate_value(r1[i],z1[i],solps_fields->vz1t[i],
                                             r2[i],z2[i],solps_fields->vz2t[i],
                                             r3[i],z3[i],solps_fields->vz3t[i],
                                             r[ii],z[jj]);
		 solps_fields->vz[index] = a_point;
		 
		 a_point = interpolate_value(r1[i],z1[i],solps_fields->Er1t[i],
                                             r2[i],z2[i],solps_fields->Er2t[i],
                                             r3[i],z3[i],solps_fields->Er3t[i],
                                             r[ii],z[jj]);
		 solps_fields->Er[index] = a_point;
		 
		 a_point = interpolate_value(r1[i],z1[i],solps_fields->Ez1t[i],
                                             r2[i],z2[i],solps_fields->Ez2t[i],
                                             r3[i],z3[i],solps_fields->Ez3t[i],
                                             r[ii],z[jj]);
		 solps_fields->Ez[index] = a_point;
		 
		 a_point = interpolate_value(r1[i],z1[i],solps_fields->gradTe1t[i],
                                             r2[i],z2[i],solps_fields->gradTe2t[i],
                                             r3[i],z3[i],solps_fields->gradTe3t[i],
                                             r[ii],z[jj]);
		 solps_fields->gradTe[index] = a_point;
		 
		 a_point = interpolate_value(r1[i],z1[i],solps_fields->gradTer1t[i],
                                             r2[i],z2[i],solps_fields->gradTer2t[i],
                                             r3[i],z3[i],solps_fields->gradTer3t[i],
                                             r[ii],z[jj]);
		 solps_fields->gradTer[index] = a_point;
		 
		 a_point = interpolate_value(r1[i],z1[i],solps_fields->gradTet1t[i],
                                             r2[i],z2[i],solps_fields->gradTet2t[i],
                                             r3[i],z3[i],solps_fields->gradTet3t[i],
                                             r[ii],z[jj]);
		 solps_fields->gradTet[index] = a_point;
		 
		 a_point = interpolate_value(r1[i],z1[i],solps_fields->gradTez1t[i],
                                             r2[i],z2[i],solps_fields->gradTez2t[i],
                                             r3[i],z3[i],solps_fields->gradTez3t[i],
                                             r[ii],z[jj]);
		 solps_fields->gradTez[index] = a_point;
		 
		 a_point = interpolate_value(r1[i],z1[i],solps_fields->gradTi1t[i],
                                             r2[i],z2[i],solps_fields->gradTi2t[i],
                                             r3[i],z3[i],solps_fields->gradTi3t[i],
                                             r[ii],z[jj]);
		 solps_fields->gradTi[index] = a_point;
		 
		 a_point = interpolate_value(r1[i],z1[i],solps_fields->gradTir1t[i],
                                             r2[i],z2[i],solps_fields->gradTir2t[i],
                                             r3[i],z3[i],solps_fields->gradTir3t[i],
                                             r[ii],z[jj]);
		 solps_fields->gradTir[index] = a_point;
		 
		 a_point = interpolate_value(r1[i],z1[i],solps_fields->gradTit1t[i],
                                             r2[i],z2[i],solps_fields->gradTit2t[i],
                                             r3[i],z3[i],solps_fields->gradTit3t[i],
                                             r[ii],z[jj]);
		 solps_fields->gradTit[index] = a_point;
		 
		 a_point = interpolate_value(r1[i],z1[i],solps_fields->gradTiz1t[i],
                                             r2[i],z2[i],solps_fields->gradTiz2t[i],
                                             r3[i],z3[i],solps_fields->gradTiz3t[i],
                                             r[ii],z[jj]);
		 solps_fields->gradTiz[index] = a_point;
               }
            }
        }
};

// Global variables

int nx, ny, ns;

std::vector<int> read_ifield(std::string filename, std::string variable_name)
{
  std::ifstream inFile;
    
    inFile.open(filename);
    if (!inFile) {
      std::cout << "Unable to open file";
        exit(1); // terminate with error
    }
    
    std::string str;
    std::size_t found;
    std::vector<int> values;

    while (getline(inFile,str)) {
        found=str.find("*cf:");
        if (found!=std::string::npos)
        {
          std::istringstream iss(str);
          std::vector<std::string> results((std::istream_iterator<std::string>(iss)),
                                   std::istream_iterator<std::string>());
          
          std::string varname = results[3];
          size_t var_name_length = variable_name.length();
          
          if (std::strncmp(varname.c_str(),variable_name.c_str(),var_name_length)== 0)
          {
            int var_count = 0;
            int var_size = 0;
            //std::sscanf(results[2].c_str(), "%d", &var_size);
            var_size = std::atoi(results[2].c_str());
            values.resize(var_size);
            while (var_count < var_size)
            {
              getline(inFile,str);
              std::istringstream iss2(str);
              std::vector<std::string> results2((std::istream_iterator<std::string>(iss2)),
                                   std::istream_iterator<std::string>());
              for (int i=0; i<results2.size();i++)
              {
                int val = 0;
                //std::sscanf(results2[i].c_str(), "%d", &val);
                val = std::atoi(results2[i].c_str());
                values[var_count] = val;
                //std::cout << val << "\n";
                var_count = var_count + 1;
              }
            }
          }
        }
    }
    
    inFile.close();
    return values;
}

std::vector<double> read_dfield(std::string filename, std::string variable_name)
{
  std::ifstream inFile;
    
    inFile.open(filename);
    if (!inFile) {
      std::cout << "Unable to open file";
        exit(1); // terminate with error
    }
    
    std::string str;
    std::size_t found;
    std::vector<double> values;

    while (getline(inFile,str)) {
        found=str.find("*cf:");
        if (found!=std::string::npos)
        {
          std::istringstream iss(str);
          std::vector<std::string> results((std::istream_iterator<std::string>(iss)),
                                   std::istream_iterator<std::string>());
          
          std::string varname = results[3];
          size_t var_name_length = varname.length();
          
          if (std::strncmp(varname.c_str(),variable_name.c_str(),var_name_length)== 0)
          {
            int var_count = 0;
            int var_size = 0;
            //std::sscanf(results[2].c_str(), "%d", &var_size);
            var_size = std::atoi(results[2].c_str());
            values.resize(var_size);
            while (var_count < var_size)
            {
              getline(inFile,str);
              std::istringstream iss2(str);
              std::vector<std::string> results2((std::istream_iterator<std::string>(iss2)),
                                   std::istream_iterator<std::string>());
              //std::cout << "results2 size " << results2.size() << std::endl;
              for (int i=0; i<results2.size();i++)
              {
                double val = 0;
                //std::sscanf(results2[i].c_str(), "%lf", &val);
                val = std::atof(results2[i].c_str());
                values[var_count] = val;
                std::cout.precision(17);
                //std::cout << results2[i] << " " << std::fixed << val <<"\n";
                var_count = var_count + 1;
              }
            }
          }
        }
    }
    
    inFile.close();
    return values;
}
int solps_2d_index(int i, int j)
{
  return j*(nx+2) + i;
}

int solps_3d_index(int i, int j, int k)
{
  return k*(nx+2)*(ny+2) + j*(nx+2) + i;
}

int solps_4d_index(int i, int j, int k, int m, int n3)
{
  return m*n3*(nx+2)*(ny+2) + k*(nx+2)*(ny+2) + j*(nx+2) + i;
}

int solps_3d_index_store(int i, int j, int k)
{
  return k*nx*ny + j*nx + i;
}

double mean(double a, double b)
{
  return 0.5*(a+b);
}

double mean(double a, double b, double c, double d)
{
  return 0.25*(a+b+c+d);
}
double cell_center(int i, int j, std::vector<double> cr)
{
  double a = cr[solps_3d_index(i,j,0)];
  double b = cr[solps_3d_index(i,j,1)];
  double c = cr[solps_3d_index(i,j,2)];
  double d = cr[solps_3d_index(i,j,3)];
  return mean(a,b,c,d);
}

double distance(double r0, double z0, double r1, double z1)
{
  return std::sqrt((r1-r0)*(r1-r0) + (z1-z0)*(z1-z0));
}

std::tuple<std::vector<double>,std::vector<double>> initialize_vectors(std::vector<double> a, std::vector<double> b)
{
  for(int m = 0;m<10;m++)
  {
    a[m] = 5;
    b[m] = 10;
  }

  return std::make_tuple(a,b);
}

std::tuple<std::vector<double>,std::vector<double>,
           std::vector<double>,std::vector<double>,
           std::vector<double>,std::vector<double>,
           std::vector<double>>
            get_triangles(std::vector<double> r1,std::vector<double> r2,
           std::vector<double> r3,std::vector<double> z1,
           std::vector<double> z2,std::vector<double> z3,
           std::vector<double> radius)
{
  std::vector<int> nxny = read_ifield("b2fgmtry","nx,ny");

  nx = nxny[0];
  ny = nxny[1];

  // Calculate Efield on SOLPS grid
  int n_total = nx*ny*8;
  r1.resize(n_total,0.0);
  r2.resize(n_total,0.0);
  r3.resize(n_total,0.0);
  z1.resize(n_total,0.0);
  z2.resize(n_total,0.0);
  z3.resize(n_total,0.0);
  radius.resize(n_total,0.0);
  
  int n_edge_total = 4*ny;
  std::vector<double> r1_edge(n_edge_total,0.0);
  std::vector<double> r2_edge(n_edge_total,0.0);
  std::vector<double> r3_edge(n_edge_total,0.0);
  std::vector<double> z1_edge(n_edge_total,0.0);
  std::vector<double> z2_edge(n_edge_total,0.0);
  std::vector<double> z3_edge(n_edge_total,0.0);
  std::vector<double> v1_edge(n_edge_total,0.0);
  std::vector<double> v2_edge(n_edge_total,0.0);
  std::vector<double> v3_edge(n_edge_total,0.0);
  std::vector<double> radius_edge(n_edge_total,0.0);

  std::cout << "SOLPS geometry: nx = " << nx << " ny = " << ny << std::endl;
    
  std::vector<double> crx = read_dfield("b2fgmtry", "crx");
  std::vector<double> cry = read_dfield("b2fgmtry", "cry");
  std::vector<double> hx = read_dfield("b2fgmtry", "hx");
  std::vector<double> hy = read_dfield("b2fgmtry", "hy");
  
  std::vector<int> leftix = read_ifield("b2fgmtry", "leftix");
  std::vector<int> leftiy = read_ifield("b2fgmtry", "leftiy");
  std::vector<int> rightix = read_ifield("b2fgmtry", "rightix");
  std::vector<int> rightiy = read_ifield("b2fgmtry", "rightiy");
  std::vector<int> topix = read_ifield("b2fgmtry", "topix");
  std::vector<int> topiy = read_ifield("b2fgmtry", "topiy");
  std::vector<int> bottomix = read_ifield("b2fgmtry", "bottomix");
  std::vector<int> bottomiy = read_ifield("b2fgmtry", "bottomiy");
    
  for (int i = 0; i< leftix.size(); i++)
          {
            leftix[i] = leftix[i] + 1;
            leftiy[i] = leftiy[i] + 1;
            rightix[i] = rightix[i] + 1;
            rightiy[i] = rightiy[i] + 1;
            topix[i] = topix[i] + 1;
            topiy[i] = topiy[i] + 1;
            bottomix[i] = bottomix[i] + 1;
            bottomiy[i] = bottomiy[i] + 1;
          }
  // Get SOLPS state variables
  std::vector<double> po = read_dfield("b2fstate", "po");

  for (int i=1; i < nx+1; i++)
  {
    for( int j=1; j < ny+1; j++)
    {
      int cell_2d_index = solps_2d_index(i,j);

      double cell_hx = hx[cell_2d_index];
      double cell_hy = hy[cell_2d_index];
      double cell_po = po[cell_2d_index];

      int top_2d_index = solps_2d_index(topix[solps_2d_index(i,j)],
                                        topiy[solps_2d_index(i,j)]);

      double top_hx = hx[top_2d_index];
      double top_hy = hy[top_2d_index];
      double top_po = po[top_2d_index];

      int bottom_2d_index = solps_2d_index(bottomix[solps_2d_index(i,j)],
                                        bottomiy[solps_2d_index(i,j)]);

      double bottom_hx = hx[bottom_2d_index];
      double bottom_hy = hy[bottom_2d_index];
      double bottom_po = po[bottom_2d_index];

      int right_2d_index = solps_2d_index(rightix[solps_2d_index(i,j)],
                                        rightiy[solps_2d_index(i,j)]);

      double right_hx = hx[right_2d_index];
      double right_hy = hy[right_2d_index];
      double right_po = po[right_2d_index];

      int left_2d_index = solps_2d_index(leftix[solps_2d_index(i,j)],
                                        leftiy[solps_2d_index(i,j)]);

      double left_hx = hx[left_2d_index];
      double left_hy = hy[left_2d_index];
      double left_po = po[left_2d_index];
      
      double r_bottom_left = crx[solps_3d_index(i,j,0)];
      double z_bottom_left = cry[solps_3d_index(i,j,0)];
      double r_bottom_right = crx[solps_3d_index(i,j,1)];
      double z_bottom_right = cry[solps_3d_index(i,j,1)];
      double r_top_left = crx[solps_3d_index(i,j,2)];
      double z_top_left = cry[solps_3d_index(i,j,2)];
      double r_top_right = crx[solps_3d_index(i,j,3)];
      double z_top_right = cry[solps_3d_index(i,j,3)];

      double r_right_mid = mean(r_top_right, r_bottom_right);
      double z_right_mid = mean(z_top_right, z_bottom_right);
      double r_left_mid = mean(r_top_left, r_bottom_left);
      double z_left_mid = mean(z_top_left, z_bottom_left);
      double r_top_mid = mean(r_top_left, r_top_right);
      double z_top_mid = mean(z_top_left, z_top_right);
      double r_bottom_mid = mean(r_bottom_left, r_bottom_right);
      double z_bottom_mid = mean(z_bottom_left, z_bottom_right);

      double norm_left_right = std::sqrt((r_right_mid - r_left_mid)*(r_right_mid - r_left_mid) +
            (z_right_mid - z_left_mid)*(z_right_mid - z_left_mid));

      double norm_bottom_top = std::sqrt((r_top_mid - r_bottom_mid)*(r_top_mid - r_bottom_mid) +
            (z_top_mid - z_bottom_mid)*(z_top_mid - z_bottom_mid));

      int i_left = leftix[solps_2d_index(i,j)];
      int j_left = leftiy[solps_2d_index(i,j)];
      int i_right = rightix[solps_2d_index(i,j)];
      int j_right = rightiy[solps_2d_index(i,j)];
      int index_left = solps_2d_index(leftix[solps_2d_index(i,j)],
                                   leftiy[solps_2d_index(i,j)]);
      int index_right = solps_2d_index(rightix[solps_2d_index(i,j)],
                                    rightiy[solps_2d_index(i,j)]);
      int index_bottom = solps_2d_index(bottomix[solps_2d_index(i,j)],
                                     bottomiy[solps_2d_index(i,j)]);
      int index_top = solps_2d_index(topix[solps_2d_index(i,j)],
                                  topiy[solps_2d_index(i,j)]);
       
      int index_topright = solps_2d_index(rightix[index_top],
                                       rightiy[index_top]);
      int index_topleft = solps_2d_index(leftix[index_top],
                                      leftiy[index_top]);
      int index_bottomright = solps_2d_index(rightix[index_bottom],
                                       rightiy[index_bottom]);
      int index_bottomleft = solps_2d_index(leftix[index_bottom],
                                      leftiy[index_bottom]);
      
      double r_cell = cell_center(i,j,crx);
      double z_cell = cell_center(i,j,cry);
      double a_cell = po[solps_2d_index(i,j)];

      double r_left = cell_center(leftix[solps_2d_index(i,j)],
                                  leftiy[solps_2d_index(i,j)],
                                  crx);
      double z_left = cell_center(leftix[solps_2d_index(i,j)],
                                  leftiy[solps_2d_index(i,j)],
                                  cry);
      double a_left = po[index_left];
      double r_right = cell_center(rightix[solps_2d_index(i,j)],
                                  rightiy[solps_2d_index(i,j)],
                                  crx);
      double z_right = cell_center(rightix[solps_2d_index(i,j)],
                                  rightiy[solps_2d_index(i,j)],
                                  cry);
      double a_right = po[index_right];
      double r_bottom = cell_center(bottomix[solps_2d_index(i,j)],
                                  bottomiy[solps_2d_index(i,j)],
                                  crx);
      double z_bottom = cell_center(bottomix[solps_2d_index(i,j)],
                                  bottomiy[solps_2d_index(i,j)],
                                  cry);
      double a_bottom = po[index_bottom];
      double r_top = cell_center(topix[solps_2d_index(i,j)],
                                  topiy[solps_2d_index(i,j)],
                                  crx);
      double z_top = cell_center(topix[solps_2d_index(i,j)],
                                  topiy[solps_2d_index(i,j)],
                                  cry);
      double a_top = po[index_top];
      double r_topright = cell_center(rightix[index_top],
                                  rightiy[index_top],
                                  crx);
      double z_topright = cell_center(rightix[index_top],
                                      rightiy[index_top],
                                  cry);
      double a_topright = po[index_topright];
      double r_topleft = cell_center(leftix[index_top],
                                     leftiy[index_top],
                                  crx);
      double z_topleft = cell_center(leftix[index_top],
                                     leftiy[index_top],
                                  cry);
      double a_topleft = po[index_topleft];
      double r_bottomright = cell_center(rightix[index_bottom],
                                  rightiy[index_bottom],
                                  crx);
      double z_bottomright = cell_center(rightix[index_bottom],
                                      rightiy[index_bottom],
                                  cry);
      double a_bottomright = po[index_bottomright];
      double r_bottomleft = cell_center(leftix[index_bottom],
                                     leftiy[index_bottom],
                                  crx);
      double z_bottomleft = cell_center(leftix[index_bottom],
                                     leftiy[index_bottom],
                                  cry);
      double a_bottomleft = po[index_bottomleft];

            //% Interpolate values at cell edges
      double d1 = distance(r_cell,z_cell,r_top_mid,z_top_mid);
      double d2 = distance(r_top_mid,z_top_mid,r_top,z_top);
      double a_top_mid = (a_top*d1 + a_cell*d2)/(d1+d2);
      d1 = distance(r_cell,z_cell,r_bottom_mid,z_bottom_mid);
      d2 = distance(r_bottom_mid,z_bottom_mid,r_bottom,z_bottom);
      double a_bottom_mid = (a_bottom*d1 + a_cell*d2)/(d1+d2);
      d1 = distance(r_cell,z_cell,r_right_mid,z_right_mid);
      d2 = distance(r_right_mid,z_right_mid,r_right,z_right);
      d1 = distance(r_cell,z_cell,r_left_mid,z_left_mid);
      d2 = distance(r_left_mid,z_left_mid,r_left,z_left);

      // % Off grid values for corners
      double r_topright_offgrid = mean(crx[solps_3d_index(i_right,j_right,2)],crx[solps_3d_index(i_right,j_right,3)]);
      double z_topright_offgrid = mean(cry[solps_3d_index(i_right,j_right,2)],cry[solps_3d_index(i_right,j_right,3)]);
      d1 = distance(r_right,z_right,r_topright_offgrid,z_topright_offgrid);
      d2 = distance(r_topright_offgrid,z_topright_offgrid,r_topright,z_topright);
      double a_topright_offgrid = (a_topright*d1 + a_right*d2)/(d1+d2);
      d1 = distance(r_top_mid,z_top_mid,r_top_right,z_top_right);
      d2 = distance(r_top_right,z_top_right,r_topright_offgrid,z_topright_offgrid);
      double a_topright_corner = (a_topright_offgrid*d1 + a_top_mid*d2)/(d1+d2);

      double r_bottomright_offgrid = mean(crx[solps_3d_index(i_right,j_right,0)],crx[solps_3d_index(i_right,j_right,1)]);
      double z_bottomright_offgrid = mean(cry[solps_3d_index(i_right,j_right,0)],cry[solps_3d_index(i_right,j_right,1)]);
      d1 = distance(r_right,z_right,r_bottomright_offgrid,z_bottomright_offgrid);
      d2 = distance(r_bottomright_offgrid,z_bottomright_offgrid,r_bottomright,z_bottomright);
      double a_bottomright_offgrid = (a_bottomright*d1 + a_right*d2)/(d1+d2);
      d1 = distance(r_bottom_mid,z_bottom_mid,r_bottom_right,z_bottom_right);
      d2 = distance(r_bottom_right,z_bottom_right,r_bottomright_offgrid,z_bottomright_offgrid);
      double a_bottomright_corner = (a_bottomright_offgrid*d1 + a_bottom_mid*d2)/(d1+d2);

      double r_topleft_offgrid = mean(crx[solps_3d_index(i_left,j_left,2)],crx[solps_3d_index(i_left,j_left,3)]);
      double z_topleft_offgrid = mean(cry[solps_3d_index(i_left,j_left,2)],cry[solps_3d_index(i_left,j_left,3)]);
      d1 = distance(r_left,z_left,r_topleft_offgrid,z_topleft_offgrid);
      d2 = distance(r_topleft_offgrid,z_topleft_offgrid,r_topleft,z_topleft);
      double a_topleft_offgrid = (a_topleft*d1 + a_left*d2)/(d1+d2);
      d1 = distance(r_top_mid,z_top_mid,r_top_left,z_top_left);
      d2 = distance(r_top_left,z_top_left,r_topleft_offgrid,z_topleft_offgrid);
      double a_topleft_corner = (a_topleft_offgrid*d1 + a_top_mid*d2)/(d1+d2);
      
      double r_bottomleft_offgrid = mean(crx[solps_3d_index(i_left,j_left,0)],crx[solps_3d_index(i_left,j_left,1)]);
      double z_bottomleft_offgrid = mean(cry[solps_3d_index(i_left,j_left,0)],cry[solps_3d_index(i_left,j_left,1)]);

      d1 = distance(r_left,z_left,r_bottomleft_offgrid,z_bottomleft_offgrid);
      d2 = distance(r_bottomleft_offgrid,z_bottomleft_offgrid,r_bottomleft,z_bottomleft);
      double a_bottomleft_offgrid = (a_bottomleft*d1 + a_left*d2)/(d1+d2);
      d1 = distance(r_bottom_mid,z_bottom_mid,r_bottom_left,z_bottom_left);
      d2 = distance(r_bottom_left,z_bottom_left,r_bottomleft_offgrid,z_bottomleft_offgrid);
      double a_bottomleft_corner = (a_bottomleft_offgrid*d1 + a_bottom_mid*d2)/(d1+d2);
      int triangle_index = solps_3d_index_store(i-1,j-1,0);
      r1[triangle_index] = r_cell;
      z1[triangle_index] = z_cell;
      //v1[triangle_index] = a_cell;
      r2[triangle_index] = r_top_mid;
      z2[triangle_index] = z_top_mid;
      //v2[triangle_index] = a_top_mid;
      r3[triangle_index] = r_top_right;
      z3[triangle_index] = z_top_right;
      //v3[triangle_index] = a_topright_corner;
      double d12 = distance(r1[triangle_index],z1[triangle_index],r2[triangle_index],z2[triangle_index]); 
      double d23 = distance(r3[triangle_index],z3[triangle_index],r2[triangle_index],z2[triangle_index]); 
      double d13 = distance(r1[triangle_index],z1[triangle_index],r3[triangle_index],z3[triangle_index]); 
      double max1 = std::max(d12,d23);
      radius[triangle_index] = std::max(max1,d13);

      triangle_index = solps_3d_index_store(i-1,j-1,1);
      r1[triangle_index] = r_cell;
      z1[triangle_index] = z_cell;
      //v1[triangle_index] = a_cell;
      r2[triangle_index] = r_top_right;
      z2[triangle_index] = z_top_right;
      //v2[triangle_index] = a_topright_corner;
      r3[triangle_index] = r_right_mid;
      z3[triangle_index] = z_right_mid;
      //v3[triangle_index] = a_right_mid;
      d12 = distance(r1[triangle_index],z1[triangle_index],r2[triangle_index],z2[triangle_index]); 
      d23 = distance(r3[triangle_index],z3[triangle_index],r2[triangle_index],z2[triangle_index]); 
      d13 = distance(r1[triangle_index],z1[triangle_index],r3[triangle_index],z3[triangle_index]); 
      max1 = std::max(d12,d23);
      radius[triangle_index] = std::max(max1,d13);
      
      triangle_index = solps_3d_index_store(i-1,j-1,2);
      r1[triangle_index] = r_cell;
      z1[triangle_index] = z_cell;
      //v1[triangle_index] = a_cell;
      r2[triangle_index] = r_right_mid;
      z2[triangle_index] = z_right_mid;
      //v2[triangle_index] = a_right_mid;
      r3[triangle_index] = r_bottom_right;
      z3[triangle_index] = z_bottom_right;
      //v3[triangle_index] = a_bottomright_corner;
      d12 = distance(r1[triangle_index],z1[triangle_index],r2[triangle_index],z2[triangle_index]); 
      d23 = distance(r3[triangle_index],z3[triangle_index],r2[triangle_index],z2[triangle_index]); 
      d13 = distance(r1[triangle_index],z1[triangle_index],r3[triangle_index],z3[triangle_index]); 
      max1 = std::max(d12,d23);
      radius[triangle_index] = std::max(max1,d13);
      
      triangle_index = solps_3d_index_store(i-1,j-1,3);
      r1[triangle_index] = r_cell;
      z1[triangle_index] = z_cell;
      //v1[triangle_index] = a_cell;
      r2[triangle_index] = r_bottom_right;
      z2[triangle_index] = z_bottom_right;
      //v2[triangle_index] = a_bottomright_corner;
      r3[triangle_index] = r_bottom_mid;
      z3[triangle_index] = z_bottom_mid;
      //v3[triangle_index] = a_bottom_mid;
      d12 = distance(r1[triangle_index],z1[triangle_index],r2[triangle_index],z2[triangle_index]); 
      d23 = distance(r3[triangle_index],z3[triangle_index],r2[triangle_index],z2[triangle_index]); 
      d13 = distance(r1[triangle_index],z1[triangle_index],r3[triangle_index],z3[triangle_index]); 
      max1 = std::max(d12,d23);
      radius[triangle_index] = std::max(max1,d13);
      
      triangle_index = solps_3d_index_store(i-1,j-1,4);
      r1[triangle_index] = r_cell;
      z1[triangle_index] = z_cell;
      //v1[triangle_index] = a_cell;
      r2[triangle_index] = r_bottom_mid;
      z2[triangle_index] = z_bottom_mid;
      //v2[triangle_index] = a_bottom_mid;
      r3[triangle_index] = r_bottom_left;
      z3[triangle_index] = z_bottom_left;
      //v3[triangle_index] = a_bottomleft_corner;
      d12 = distance(r1[triangle_index],z1[triangle_index],r2[triangle_index],z2[triangle_index]); 
      d23 = distance(r3[triangle_index],z3[triangle_index],r2[triangle_index],z2[triangle_index]); 
      d13 = distance(r1[triangle_index],z1[triangle_index],r3[triangle_index],z3[triangle_index]); 
      max1 = std::max(d12,d23);
      radius[triangle_index] = std::max(max1,d13);
      
      triangle_index = solps_3d_index_store(i-1,j-1,5);
      r1[triangle_index] = r_cell;
      z1[triangle_index] = z_cell;
      //v1[triangle_index] = a_cell;
      r2[triangle_index] = r_bottom_left;
      z2[triangle_index] = z_bottom_left;
      //v2[triangle_index] = a_bottomleft_corner;
      r3[triangle_index] = r_left_mid;
      z3[triangle_index] = z_left_mid;
      //v3[triangle_index] = a_left_mid;
      d12 = distance(r1[triangle_index],z1[triangle_index],r2[triangle_index],z2[triangle_index]); 
      d23 = distance(r3[triangle_index],z3[triangle_index],r2[triangle_index],z2[triangle_index]); 
      d13 = distance(r1[triangle_index],z1[triangle_index],r3[triangle_index],z3[triangle_index]); 
      max1 = std::max(d12,d23);
      radius[triangle_index] = std::max(max1,d13);
      
      triangle_index = solps_3d_index_store(i-1,j-1,6);
      r1[triangle_index] = r_cell;
      z1[triangle_index] = z_cell;
      //v1[triangle_index] = a_cell;
      r2[triangle_index] = r_left_mid;
      z2[triangle_index] = z_left_mid;
      //v2[triangle_index] = a_left_mid;
      r3[triangle_index] = r_top_left;
      z3[triangle_index] = z_top_left;
      //v3[triangle_index] = a_topleft_corner;
      d12 = distance(r1[triangle_index],z1[triangle_index],r2[triangle_index],z2[triangle_index]); 
      d23 = distance(r3[triangle_index],z3[triangle_index],r2[triangle_index],z2[triangle_index]); 
      d13 = distance(r1[triangle_index],z1[triangle_index],r3[triangle_index],z3[triangle_index]); 
      max1 = std::max(d12,d23);
      radius[triangle_index] = std::max(max1,d13);
      
      triangle_index = solps_3d_index_store(i-1,j-1,7);
      r1[triangle_index] = r_cell;
      z1[triangle_index] = z_cell;
      //v1[triangle_index] = a_cell;
      r2[triangle_index] = r_top_left;
      z2[triangle_index] = z_top_left;
      //v2[triangle_index] = a_topleft_corner;
      r3[triangle_index] = r_top_mid;
      z3[triangle_index] = z_top_mid;
      //v3[triangle_index] = a_top_mid;
      d12 = distance(r1[triangle_index],z1[triangle_index],r2[triangle_index],z2[triangle_index]); 
      d23 = distance(r3[triangle_index],z3[triangle_index],r2[triangle_index],z2[triangle_index]); 
      d13 = distance(r1[triangle_index],z1[triangle_index],r3[triangle_index],z3[triangle_index]); 
      max1 = std::max(d12,d23);
      radius[triangle_index] = std::max(max1,d13);

      if(i==1)
      {
	        double target_buff = 0.01;
		r1_edge[(j-1)*4 + 0] = r_top_left;
		z1_edge[(j-1)*4 + 0] = z_top_left;
		r2_edge[(j-1)*4 + 0] = r_bottom_left;
		z2_edge[(j-1)*4 + 0] = z_bottom_left;
		v1_edge[(j-1)*4 + 0] = a_topleft_corner;
		v2_edge[(j-1)*4 + 0] = a_bottomleft_corner;

		r1_edge[(j-1)*4 + 1] = r_bottom_left;
		z1_edge[(j-1)*4 + 1] = z_bottom_left;
		v1_edge[(j-1)*4 + 1] = a_bottomleft_corner;


                double parvecr = r_bottom_left - r_top_left;
		double parvecz = r_bottom_left - z_top_left;
                double parvecnorm = std::sqrt(parvecr*parvecr + parvecz*parvecz);
		parvecr = parvecr/parvecnorm;
		parvecz = parvecz/parvecnorm;

                double perpvecr = -parvecz;
		double perpvecz = parvecr;
                
		r2_edge[(j-1)*4 + 1] = r_top_left + target_buff*perpvecr;
		z2_edge[(j-1)*4 + 1] = z_top_left + target_buff*perpvecz;
		v2_edge[(j-1)*4 + 1] = a_topleft_corner;
		
		r3_edge[(j-1)*4 + 0] = r_top_left + target_buff*perpvecr;
		r3_edge[(j-1)*4 + 1] = r_bottom_left + target_buff*perpvecr;
		z3_edge[(j-1)*4 + 0] = z_top_left + target_buff*perpvecz;
		z3_edge[(j-1)*4 + 1] = z_bottom_left + target_buff*perpvecz;
		v3_edge[(j-1)*4 + 0] = a_topleft_corner;
		v3_edge[(j-1)*4 + 1] = a_bottomleft_corner;
      
		d12 = distance(r1_edge[(j-1)*4 + 0],z1_edge[(j-1)*4 + 0],
			       r2_edge[(j-1)*4 + 0],z2_edge[(j-1)*4 + 0]); 
		d23 = distance(r3_edge[(j-1)*4 + 0],z3_edge[(j-1)*4 + 0],
			       r2_edge[(j-1)*4 + 0],z2_edge[(j-1)*4 + 0]); 
		d13 = distance(r1_edge[(j-1)*4 + 0],z1_edge[(j-1)*4 + 0],
			       r3_edge[(j-1)*4 + 0],z3_edge[(j-1)*4 + 0]); 
      		max1 = std::max(d12,d23);
      		radius_edge[(j-1)*4 + 0] = std::max(max1,d13);
		
		d12 = distance(r1_edge[(j-1)*4 + 1],z1_edge[(j-1)*4 + 1],
			       r2_edge[(j-1)*4 + 1],z2_edge[(j-1)*4 + 1]); 
		d23 = distance(r3_edge[(j-1)*4 + 1],z3_edge[(j-1)*4 + 1],
			       r2_edge[(j-1)*4 + 1],z2_edge[(j-1)*4 + 1]); 
		d13 = distance(r1_edge[(j-1)*4 + 1],z1_edge[(j-1)*4 + 1],
			       r3_edge[(j-1)*4 + 1],z3_edge[(j-1)*4 + 1]); 
      		max1 = std::max(d12,d23);
      		radius_edge[(j-1)*4 + 1] = std::max(max1,d13);
      }
      
      if(i==nx)
      {
	        double target_buff = 0.01;
		r1_edge[(j-1)*4 + 2] = r_top_right;
		z1_edge[(j-1)*4 + 2] = z_top_right;
		r2_edge[(j-1)*4 + 2] = r_bottom_right;
		z2_edge[(j-1)*4 + 2] = z_bottom_right;
		v1_edge[(j-1)*4 + 2] = a_topright_corner;
		v2_edge[(j-1)*4 + 2] = a_bottomright_corner;

		r1_edge[(j-1)*4 + 3] = r_bottom_right;
		z1_edge[(j-1)*4 + 3] = z_bottom_right;
		v1_edge[(j-1)*4 + 3] = a_bottomright_corner;

                double parvecr = r_bottom_right - r_top_right;
		double parvecz = r_bottom_right - z_top_right;
                double parvecnorm = std::sqrt(parvecr*parvecr + parvecz*parvecz);
		parvecr = parvecr/parvecnorm;
		parvecz = parvecz/parvecnorm;

                double perpvecr = -(-parvecz);
		double perpvecz = -parvecr;

		r2_edge[(j-1)*4 + 3] = r_top_right + target_buff*perpvecr;
		z2_edge[(j-1)*4 + 3] = z_top_right + target_buff*perpvecz;
		v2_edge[(j-1)*4 + 3] = a_topright_corner;
		
		r3_edge[(j-1)*4 + 2] = r_top_right + target_buff*perpvecr;
		r3_edge[(j-1)*4 + 3] = r_bottom_right + target_buff*perpvecr;
		z3_edge[(j-1)*4 + 2] = z_top_right + target_buff*perpvecz;
		z3_edge[(j-1)*4 + 3] = z_bottom_right + target_buff*perpvecz;
		v3_edge[(j-1)*4 + 2] = a_topright_corner;
		v3_edge[(j-1)*4 + 3] = a_bottomright_corner;
      
		d12 = distance(r1_edge[(j-1)*4 + 2],z1_edge[(j-1)*4 + 2],
			       r2_edge[(j-1)*4 + 2],z2_edge[(j-1)*4 + 2]); 
		d23 = distance(r3_edge[(j-1)*4 + 2],z3_edge[(j-1)*4 + 2],
			       r2_edge[(j-1)*4 + 2],z2_edge[(j-1)*4 + 2]); 
		d13 = distance(r1_edge[(j-1)*4 + 2],z1_edge[(j-1)*4 + 2],
			       r3_edge[(j-1)*4 + 2],z3_edge[(j-1)*4 + 2]); 
      		max1 = std::max(d12,d23);
      		radius_edge[(j-1)*4 + 2] = std::max(max1,d13);
		
		d12 = distance(r1_edge[(j-1)*4 + 3],z1_edge[(j-1)*4 + 3],
			       r2_edge[(j-1)*4 + 3],z2_edge[(j-1)*4 + 3]); 
		d23 = distance(r3_edge[(j-1)*4 + 3],z3_edge[(j-1)*4 + 3],
			       r2_edge[(j-1)*4 + 3],z2_edge[(j-1)*4 + 3]); 
		d13 = distance(r1_edge[(j-1)*4 + 3],z1_edge[(j-1)*4 + 3],
			       r3_edge[(j-1)*4 + 3],z3_edge[(j-1)*4 + 3]); 
      		max1 = std::max(d12,d23);
      		radius_edge[(j-1)*4 + 3] = std::max(max1,d13);
      }

    }
  }

  for(int ii = 0; ii< 4*ny;ii++)
  {
	  r1.push_back(r1_edge[ii]);
	  r2.push_back(r2_edge[ii]);
	  r3.push_back(r3_edge[ii]);
	  z1.push_back(z1_edge[ii]);
	  z2.push_back(z2_edge[ii]);
	  z3.push_back(z3_edge[ii]);
	  radius.push_back(radius_edge[ii]);
  }
  return std::make_tuple(r1, r2, r3, z1, z2, z3, radius);
}
std::tuple<std::vector<double>,std::vector<double>>
            get_Efield(std::vector<double> Er, std::vector<double> Ez)
{
  std::vector<int> nxny = read_ifield("b2fgmtry","nx,ny");

  nx = nxny[0];
  ny = nxny[1];

  // Calculate Efield on SOLPS grid
  Er.resize((nx+2)*(ny+2),0.0);
  Ez.resize((nx+2)*(ny+2),0.0);
  
  int n_edge_total = 4*ny;
  std::vector<double> Er1_edge(n_edge_total,0.0);
  std::vector<double> Er2_edge(n_edge_total,0.0);
  std::vector<double> Er3_edge(n_edge_total,0.0);
  std::vector<double> Ez1_edge(n_edge_total,0.0);
  std::vector<double> Ez2_edge(n_edge_total,0.0);
  std::vector<double> Ez3_edge(n_edge_total,0.0);
  std::vector<double> radius_edge(n_edge_total,0.0);
  
  std::vector<double> crx = read_dfield("b2fgmtry", "crx");
  std::vector<double> cry = read_dfield("b2fgmtry", "cry");
  std::vector<double> hx = read_dfield("b2fgmtry", "hx");
  std::vector<double> hy = read_dfield("b2fgmtry", "hy");
  
  std::vector<int> leftix = read_ifield("b2fgmtry", "leftix");
  std::vector<int> leftiy = read_ifield("b2fgmtry", "leftiy");
  std::vector<int> rightix = read_ifield("b2fgmtry", "rightix");
  std::vector<int> rightiy = read_ifield("b2fgmtry", "rightiy");
  std::vector<int> topix = read_ifield("b2fgmtry", "topix");
  std::vector<int> topiy = read_ifield("b2fgmtry", "topiy");
  std::vector<int> bottomix = read_ifield("b2fgmtry", "bottomix");
  std::vector<int> bottomiy = read_ifield("b2fgmtry", "bottomiy");
    
  for (int i = 0; i< leftix.size(); i++)
          {
            leftix[i] = leftix[i] + 1;
            leftiy[i] = leftiy[i] + 1;
            rightix[i] = rightix[i] + 1;
            rightiy[i] = rightiy[i] + 1;
            topix[i] = topix[i] + 1;
            topiy[i] = topiy[i] + 1;
            bottomix[i] = bottomix[i] + 1;
            bottomiy[i] = bottomiy[i] + 1;
          }
  // Get SOLPS state variables
  std::vector<double> po = read_dfield("b2fstate", "po");

  for (int i=1; i < nx+1; i++)
  {
    for( int j=1; j < ny+1; j++)
    {
      int cell_2d_index = solps_2d_index(i,j);

      double cell_hx = hx[cell_2d_index];
      double cell_hy = hy[cell_2d_index];
      double cell_po = po[cell_2d_index];

      int top_2d_index = solps_2d_index(topix[solps_2d_index(i,j)],
                                        topiy[solps_2d_index(i,j)]);

      double top_hx = hx[top_2d_index];
      double top_hy = hy[top_2d_index];
      double top_po = po[top_2d_index];

      int bottom_2d_index = solps_2d_index(bottomix[solps_2d_index(i,j)],
                                        bottomiy[solps_2d_index(i,j)]);

      double bottom_hx = hx[bottom_2d_index];
      double bottom_hy = hy[bottom_2d_index];
      double bottom_po = po[bottom_2d_index];

      int right_2d_index = solps_2d_index(rightix[solps_2d_index(i,j)],
                                        rightiy[solps_2d_index(i,j)]);

      double right_hx = hx[right_2d_index];
      double right_hy = hy[right_2d_index];
      double right_po = po[right_2d_index];

      int left_2d_index = solps_2d_index(leftix[solps_2d_index(i,j)],
                                        leftiy[solps_2d_index(i,j)]);

      double left_hx = hx[left_2d_index];
      double left_hy = hy[left_2d_index];
      double left_po = po[left_2d_index];
      
      double d_bottom_top = 0.5*top_hy + cell_hy + 0.5*bottom_hy;
      double d_left_right = 0.5*left_hx + cell_hx + 0.5*right_hx;

      double r_bottom_left = crx[solps_3d_index(i,j,0)];
      double z_bottom_left = cry[solps_3d_index(i,j,0)];
      double r_bottom_right = crx[solps_3d_index(i,j,1)];
      double z_bottom_right = cry[solps_3d_index(i,j,1)];
      double r_top_left = crx[solps_3d_index(i,j,2)];
      double z_top_left = cry[solps_3d_index(i,j,2)];
      double r_top_right = crx[solps_3d_index(i,j,3)];
      double z_top_right = cry[solps_3d_index(i,j,3)];

      double r_right_mid = mean(r_top_right, r_bottom_right);
      double z_right_mid = mean(z_top_right, z_bottom_right);
      double r_left_mid = mean(r_top_left, r_bottom_left);
      double z_left_mid = mean(z_top_left, z_bottom_left);
      double r_top_mid = mean(r_top_left, r_top_right);
      double z_top_mid = mean(z_top_left, z_top_right);
      double r_bottom_mid = mean(r_bottom_left, r_bottom_right);
      double z_bottom_mid = mean(z_bottom_left, z_bottom_right);

      double norm_left_right = std::sqrt((r_right_mid - r_left_mid)*(r_right_mid - r_left_mid) +
            (z_right_mid - z_left_mid)*(z_right_mid - z_left_mid));
      double r_hatx = (r_right_mid - r_left_mid)/norm_left_right;
      double z_hatx = (z_right_mid - z_left_mid)/norm_left_right;

      double dx = (right_po - left_po)/d_left_right;
      double dxr = dx*r_hatx;
      double dxz = dx*z_hatx;

      double norm_bottom_top = std::sqrt((r_top_mid - r_bottom_mid)*(r_top_mid - r_bottom_mid) +
            (z_top_mid - z_bottom_mid)*(z_top_mid - z_bottom_mid));
      double r_haty = (r_top_mid - r_bottom_mid)/norm_bottom_top;
      double z_haty = (z_top_mid - z_bottom_mid)/norm_bottom_top;


      double dy = (top_po - bottom_po)/d_bottom_top;
      double dyr = dy*r_haty;
      double dyz = dy*z_haty;
      double der = dxr+dyr;
      double dez = dxz+dyz;

      Er[cell_2d_index] = -der;
      Ez[cell_2d_index] = -dez;

      if( i == 1)
      {
        cell_2d_index = solps_2d_index(i-1,j);
	Er[cell_2d_index] = -der;
	Ez[cell_2d_index] = -dez;
      }

      if( i == nx)
      {
        cell_2d_index = solps_2d_index(i+1,j);
	Er[cell_2d_index] = -der;
	Ez[cell_2d_index] = -dez;
      }
    }
  }

  return std::make_tuple(Er, Ez);
}

std::tuple<std::vector<double>,std::vector<double>>
            get_gradT(std::vector<double> gradTe, std::vector<double> gradTi)
{
  std::vector<int> nxny = read_ifield("b2fgmtry","nx,ny");

  nx = nxny[0];
  ny = nxny[1];

  // Calculate Efield on SOLPS grid
  gradTe.resize((nx+2)*(ny+2),0.0);
  gradTi.resize((nx+2)*(ny+2),0.0);
  
  std::vector<double> crx = read_dfield("b2fgmtry", "crx");
  std::vector<double> cry = read_dfield("b2fgmtry", "cry");
  std::vector<double> hx = read_dfield("b2fgmtry", "hx");
  std::vector<double> hy = read_dfield("b2fgmtry", "hy");
  
  std::vector<int> leftix = read_ifield("b2fgmtry", "leftix");
  std::vector<int> leftiy = read_ifield("b2fgmtry", "leftiy");
  std::vector<int> rightix = read_ifield("b2fgmtry", "rightix");
  std::vector<int> rightiy = read_ifield("b2fgmtry", "rightiy");
  std::vector<int> topix = read_ifield("b2fgmtry", "topix");
  std::vector<int> topiy = read_ifield("b2fgmtry", "topiy");
  std::vector<int> bottomix = read_ifield("b2fgmtry", "bottomix");
  std::vector<int> bottomiy = read_ifield("b2fgmtry", "bottomiy");
    
  for (int i = 0; i< leftix.size(); i++)
          {
            leftix[i] = leftix[i] + 1;
            leftiy[i] = leftiy[i] + 1;
            rightix[i] = rightix[i] + 1;
            rightiy[i] = rightiy[i] + 1;
            topix[i] = topix[i] + 1;
            topiy[i] = topiy[i] + 1;
            bottomix[i] = bottomix[i] + 1;
            bottomiy[i] = bottomiy[i] + 1;
          }
  // Get SOLPS state variables
  std::vector<double> te = read_dfield("b2fstate", "te");
  std::vector<double> ti = read_dfield("b2fstate", "ti");
  std::vector<double> bb = read_dfield("b2fgmtry", "bb");
    for(int i=0; i<ti.size();i++)
    {
      ti[i] = ti[i]/1.602176565e-19;
      te[i] = te[i]/1.602176565e-19;
    }

  for (int i=1; i < nx+1; i++)
  {
    for( int j=1; j < ny+1; j++)
    {
      int cell_2d_index = solps_2d_index(i,j);

      double cell_hx = hx[cell_2d_index];
      double cell_te = te[cell_2d_index];
      double cell_ti = ti[cell_2d_index];

      int right_2d_index = solps_2d_index(rightix[solps_2d_index(i,j)],
                                        rightiy[solps_2d_index(i,j)]);

      double right_hx = hx[right_2d_index];
      double right_te = te[right_2d_index];
      double right_ti = ti[right_2d_index];

      int left_2d_index = solps_2d_index(leftix[solps_2d_index(i,j)],
                                        leftiy[solps_2d_index(i,j)]);

      double left_hx = hx[left_2d_index];
      double left_te = te[left_2d_index];
      double left_ti = ti[left_2d_index];
      
      double d_left_right = 0.5*left_hx + cell_hx + 0.5*right_hx;
      double theta = std::atan(bb[solps_3d_index(i,j,0)]/bb[solps_3d_index(i,j,2)]);
      double d_grad = d_left_right/std::sin(theta);
      //std::cout << " dlr dgrag " << d_left_right << " " << d_grad << std::endl;
      //std::cout << " rte lte " << right_te-left_te <<  std::endl;
      gradTe[cell_2d_index] = (right_te - left_te)/d_grad;
      gradTi[cell_2d_index] = (right_ti - left_ti)/d_grad;
      
      if( i == 1)
      {
        cell_2d_index = solps_2d_index(i-1,j);
	gradTe[cell_2d_index] = (right_te - left_te)/d_grad;
	gradTi[cell_2d_index] = (right_ti - left_ti)/d_grad;;
      }

      if( i == nx)
      {
        cell_2d_index = solps_2d_index(i+1,j);
	gradTe[cell_2d_index] = (right_te - left_te)/d_grad;
	gradTi[cell_2d_index] = (right_ti - left_ti)/d_grad;;
      }
    }
  }

  return std::make_tuple(gradTe, gradTi);
}

std::tuple<std::vector<double>,std::vector<double>,std::vector<double>,std::vector<double>>
            get_Bfield(std::vector<double> Br, std::vector<double> Bt, std::vector<double> Bz, 
			    std::vector<double> Bmag)
{
  std::vector<int> nxny = read_ifield("b2fgmtry","nx,ny");

  nx = nxny[0];
  ny = nxny[1];

  // Calculate Efield on SOLPS grid
  Br.resize((nx+2)*(ny+2),0.0);
  Bt.resize((nx+2)*(ny+2),0.0);
  Bz.resize((nx+2)*(ny+2),0.0);
  Bmag.resize((nx+2)*(ny+2),0.0);
  
  std::vector<double> crx = read_dfield("b2fgmtry", "crx");
  std::vector<double> cry = read_dfield("b2fgmtry", "cry");
  //std::vector<double> hx = read_dfield("b2fgmtry", "hx");
  //std::vector<double> hy = read_dfield("b2fgmtry", "hy");
  
  //std::vector<int> leftix = read_ifield("b2fgmtry", "leftix");
  //std::vector<int> leftiy = read_ifield("b2fgmtry", "leftiy");
  //std::vector<int> rightix = read_ifield("b2fgmtry", "rightix");
  //std::vector<int> rightiy = read_ifield("b2fgmtry", "rightiy");
  //std::vector<int> topix = read_ifield("b2fgmtry", "topix");
  //std::vector<int> topiy = read_ifield("b2fgmtry", "topiy");
  //std::vector<int> bottomix = read_ifield("b2fgmtry", "bottomix");
  //std::vector<int> bottomiy = read_ifield("b2fgmtry", "bottomiy");
    
  //for (int i = 0; i< leftix.size(); i++)
  //        {
  //          leftix[i] = leftix[i] + 1;
  //          leftiy[i] = leftiy[i] + 1;
  //          rightix[i] = rightix[i] + 1;
  //          rightiy[i] = rightiy[i] + 1;
  //          topix[i] = topix[i] + 1;
  //          topiy[i] = topiy[i] + 1;
  //          bottomix[i] = bottomix[i] + 1;
  //          bottomiy[i] = bottomiy[i] + 1;
  //        }
  // Get SOLPS state variables
  std::vector<double> bb = read_dfield("b2fgmtry", "bb");
  std::vector<std::vector<double>> bfield(4);
    
  for ( int i = 0 ; i < 4 ; i++ )
  {
    bfield[i].resize((nx+2)*(ny+2));
  }

  for (int i=0; i < nx+2; i++)
  {
    for( int j=0; j < ny+2; j++)
    {
      int cell_2d_index = solps_2d_index(i,j);
	
        bfield[0][cell_2d_index] = bb[solps_3d_index(i,j,0)];	
        bfield[1][cell_2d_index] = bb[solps_3d_index(i,j,1)];	
        bfield[2][cell_2d_index] = -bb[solps_3d_index(i,j,2)];	// Solps toroidal field is out of page
	//this is opposite to the GITR right handed coordinate system
        bfield[3][cell_2d_index] = bb[solps_3d_index(i,j,3)];

      //double cell_hx = hx[cell_2d_index];
      //double cell_hy = hy[cell_2d_index];
      //double cell_po = po[cell_2d_index];

      //int top_2d_index = solps_2d_index(topix[solps_2d_index(i,j)],
      //                                  topiy[solps_2d_index(i,j)]);

      //double top_hx = hx[top_2d_index];
      //double top_hy = hy[top_2d_index];
      //double top_po = po[top_2d_index];

      //int bottom_2d_index = solps_2d_index(bottomix[solps_2d_index(i,j)],
      //                                  bottomiy[solps_2d_index(i,j)]);

      //double bottom_hx = hx[bottom_2d_index];
      //double bottom_hy = hy[bottom_2d_index];
      //double bottom_po = po[bottom_2d_index];

      //int right_2d_index = solps_2d_index(rightix[solps_2d_index(i,j)],
      //                                  rightiy[solps_2d_index(i,j)]);

      //double right_hx = hx[right_2d_index];
      //double right_hy = hy[right_2d_index];
      //double right_po = po[right_2d_index];

      //int left_2d_index = solps_2d_index(leftix[solps_2d_index(i,j)],
      //                                  leftiy[solps_2d_index(i,j)]);

      //double left_hx = hx[left_2d_index];
      //double left_hy = hy[left_2d_index];
      //double left_po = po[left_2d_index];
      //
      //double d_bottom_top = 0.5*top_hy + cell_hy + 0.5*bottom_hy;
      //double d_left_right = 0.5*left_hx + cell_hx + 0.5*right_hx;

      double r_bottom_left = crx[solps_3d_index(i,j,0)];
      double z_bottom_left = cry[solps_3d_index(i,j,0)];
      double r_bottom_right = crx[solps_3d_index(i,j,1)];
      double z_bottom_right = cry[solps_3d_index(i,j,1)];
      double r_top_left = crx[solps_3d_index(i,j,2)];
      double z_top_left = cry[solps_3d_index(i,j,2)];
      double r_top_right = crx[solps_3d_index(i,j,3)];
      double z_top_right = cry[solps_3d_index(i,j,3)];

      double r_right_mid = mean(r_top_right, r_bottom_right);
      double z_right_mid = mean(z_top_right, z_bottom_right);
      double r_left_mid = mean(r_top_left, r_bottom_left);
      double z_left_mid = mean(z_top_left, z_bottom_left);
      //double r_top_mid = mean(r_top_left, r_top_right);
      //double z_top_mid = mean(z_top_left, z_top_right);
      //double r_bottom_mid = mean(r_bottom_left, r_bottom_right);
      //double z_bottom_mid = mean(z_bottom_left, z_bottom_right);

      double norm_left_right = std::sqrt((r_right_mid - r_left_mid)*(r_right_mid - r_left_mid) +
            (z_right_mid - z_left_mid)*(z_right_mid - z_left_mid));
      double r_hatx = (r_right_mid - r_left_mid)/norm_left_right;
      double z_hatx = (z_right_mid - z_left_mid)/norm_left_right;

      //double dx = (right_po - left_po)/d_left_right;
      Br[cell_2d_index] = bfield[0][cell_2d_index]*r_hatx;	
      Bt[cell_2d_index] = bfield[2][cell_2d_index];	
      Bz[cell_2d_index] = bfield[0][cell_2d_index]*z_hatx;	
      Bmag[cell_2d_index] = bfield[3][cell_2d_index];	
      //std::cout << " Bmag " << std::sqrt(bfield[0][cell_2d_index]*bfield[0][cell_2d_index] 
      //	      + bfield[2][cell_2d_index]*bfield[2][cell_2d_index]) << " " << Bmag[cell_2d_index] << std::endl;
      //double dxr = dx*r_hatx;
      //double dxz = dx*z_hatx;

      //double norm_bottom_top = std::sqrt((r_top_mid - r_bottom_mid)*(r_top_mid - r_bottom_mid) +
      //      (z_top_mid - z_bottom_mid)*(z_top_mid - z_bottom_mid));
      //double r_haty = (r_top_mid - r_bottom_mid)/norm_bottom_top;
      //double z_haty = (z_top_mid - z_bottom_mid)/norm_bottom_top;


      //double dy = (top_po - bottom_po)/d_bottom_top;
      //double dyr = dy*r_haty;
      //double dyz = dy*z_haty;
      //double der = dxr+dyr;
      //double dez = dxz+dyz;

    }
  }

  return std::make_tuple(Br, Bt, Bz, Bmag);
}
//std::tuple<std::vector<double>,std::vector<double>,std::vector<double>>
//            get_scalar_field(std::string scalar_string,
//           std::vector<double> v1,std::vector<double> v2,
//           std::vector<double> v3)
//{
//           std::vector<double> r1, r2, r3, z1, z2, z3, radius;
//  
//           std::vector<int> nxny = read_ifield("b2fgmtry","nx,ny");
//
//  nx = nxny[0];
//  ny = nxny[1];
//
//  // Calculate Efield on SOLPS grid
//  int n_total = nx*ny*8;
//  r1.resize(n_total,0.0);
//  r2.resize(n_total,0.0);
//  r3.resize(n_total,0.0);
//  z1.resize(n_total,0.0);
//  z2.resize(n_total,0.0);
//  z3.resize(n_total,0.0);
//  v1.resize(n_total,0.0);
//  v2.resize(n_total,0.0);
//  v3.resize(n_total,0.0);
//  radius.resize(n_total,0.0);
//  
//  int n_edge_total = 2*ny;
//  std::vector<double> r1_edge(n_edge_total,0.0);
//  std::vector<double> r2_edge(n_edge_total,0.0);
//  std::vector<double> r3_edge(n_edge_total,0.0);
//  std::vector<double> z1_edge(n_edge_total,0.0);
//  std::vector<double> z2_edge(n_edge_total,0.0);
//  std::vector<double> z3_edge(n_edge_total,0.0);
//  std::vector<double> v1_edge(n_edge_total,0.0);
//  std::vector<double> v2_edge(n_edge_total,0.0);
//  std::vector<double> v3_edge(n_edge_total,0.0);
//  std::vector<double> radius_edge(n_edge_total,0.0);
//    
//  std::vector<double> crx = read_dfield("b2fgmtry", "crx");
//  std::vector<double> cry = read_dfield("b2fgmtry", "cry");
//  std::vector<double> hx = read_dfield("b2fgmtry", "hx");
//  std::vector<double> hy = read_dfield("b2fgmtry", "hy");
//  
//  std::vector<int> leftix = read_ifield("b2fgmtry", "leftix");
//  std::vector<int> leftiy = read_ifield("b2fgmtry", "leftiy");
//  std::vector<int> rightix = read_ifield("b2fgmtry", "rightix");
//  std::vector<int> rightiy = read_ifield("b2fgmtry", "rightiy");
//  std::vector<int> topix = read_ifield("b2fgmtry", "topix");
//  std::vector<int> topiy = read_ifield("b2fgmtry", "topiy");
//  std::vector<int> bottomix = read_ifield("b2fgmtry", "bottomix");
//  std::vector<int> bottomiy = read_ifield("b2fgmtry", "bottomiy");
//    
//  for (int i = 0; i< leftix.size(); i++)
//          {
//            leftix[i] = leftix[i] + 1;
//            leftiy[i] = leftiy[i] + 1;
//            rightix[i] = rightix[i] + 1;
//            rightiy[i] = rightiy[i] + 1;
//            topix[i] = topix[i] + 1;
//            topiy[i] = topiy[i] + 1;
//            bottomix[i] = bottomix[i] + 1;
//            bottomiy[i] = bottomiy[i] + 1;
//          }
//  // Get SOLPS state variables
//  std::vector<double> po = read_dfield("b2fstate", scalar_string);
//
//  for (int i=1; i < nx+1; i++)
//  {
//    for( int j=1; j < ny+1; j++)
//    {
//      int cell_2d_index = solps_2d_index(i,j);
//
//      double cell_hx = hx[cell_2d_index];
//      double cell_hy = hy[cell_2d_index];
//      double cell_po = po[cell_2d_index];
//
//      int top_2d_index = solps_2d_index(topix[solps_2d_index(i,j)],
//                                        topiy[solps_2d_index(i,j)]);
//
//      double top_hx = hx[top_2d_index];
//      double top_hy = hy[top_2d_index];
//      double top_po = po[top_2d_index];
//
//      int bottom_2d_index = solps_2d_index(bottomix[solps_2d_index(i,j)],
//                                        bottomiy[solps_2d_index(i,j)]);
//
//      double bottom_hx = hx[bottom_2d_index];
//      double bottom_hy = hy[bottom_2d_index];
//      double bottom_po = po[bottom_2d_index];
//
//      int right_2d_index = solps_2d_index(rightix[solps_2d_index(i,j)],
//                                        rightiy[solps_2d_index(i,j)]);
//
//      double right_hx = hx[right_2d_index];
//      double right_hy = hy[right_2d_index];
//      double right_po = po[right_2d_index];
//
//      int left_2d_index = solps_2d_index(leftix[solps_2d_index(i,j)],
//                                        leftiy[solps_2d_index(i,j)]);
//
//      double left_hx = hx[left_2d_index];
//      double left_hy = hy[left_2d_index];
//      double left_po = po[left_2d_index];
//      
//      double r_bottom_left = crx[solps_3d_index(i,j,0)];
//      double z_bottom_left = cry[solps_3d_index(i,j,0)];
//      double r_bottom_right = crx[solps_3d_index(i,j,1)];
//      double z_bottom_right = cry[solps_3d_index(i,j,1)];
//      double r_top_left = crx[solps_3d_index(i,j,2)];
//      double z_top_left = cry[solps_3d_index(i,j,2)];
//      double r_top_right = crx[solps_3d_index(i,j,3)];
//      double z_top_right = cry[solps_3d_index(i,j,3)];
//
//      double r_right_mid = mean(r_top_right, r_bottom_right);
//      double z_right_mid = mean(z_top_right, z_bottom_right);
//      double r_left_mid = mean(r_top_left, r_bottom_left);
//      double z_left_mid = mean(z_top_left, z_bottom_left);
//      double r_top_mid = mean(r_top_left, r_top_right);
//      double z_top_mid = mean(z_top_left, z_top_right);
//      double r_bottom_mid = mean(r_bottom_left, r_bottom_right);
//      double z_bottom_mid = mean(z_bottom_left, z_bottom_right);
//
//      double norm_left_right = std::sqrt((r_right_mid - r_left_mid)*(r_right_mid - r_left_mid) +
//            (z_right_mid - z_left_mid)*(z_right_mid - z_left_mid));
//
//      double norm_bottom_top = std::sqrt((r_top_mid - r_bottom_mid)*(r_top_mid - r_bottom_mid) +
//            (z_top_mid - z_bottom_mid)*(z_top_mid - z_bottom_mid));
//
//      int i_left = leftix[solps_2d_index(i,j)];
//      int j_left = leftiy[solps_2d_index(i,j)];
//      int i_right = rightix[solps_2d_index(i,j)];
//      int j_right = rightiy[solps_2d_index(i,j)];
//      int index_left = solps_2d_index(leftix[solps_2d_index(i,j)],
//                                   leftiy[solps_2d_index(i,j)]);
//      int index_right = solps_2d_index(rightix[solps_2d_index(i,j)],
//                                    rightiy[solps_2d_index(i,j)]);
//      int index_bottom = solps_2d_index(bottomix[solps_2d_index(i,j)],
//                                     bottomiy[solps_2d_index(i,j)]);
//      int index_top = solps_2d_index(topix[solps_2d_index(i,j)],
//                                  topiy[solps_2d_index(i,j)]);
//       
//      int index_topright = solps_2d_index(rightix[index_top],
//                                       rightiy[index_top]);
//      int index_topleft = solps_2d_index(leftix[index_top],
//                                      leftiy[index_top]);
//      int index_bottomright = solps_2d_index(rightix[index_bottom],
//                                       rightiy[index_bottom]);
//      int index_bottomleft = solps_2d_index(leftix[index_bottom],
//                                      leftiy[index_bottom]);
//      
//      double r_cell = cell_center(i,j,crx);
//      double z_cell = cell_center(i,j,cry);
//      double a_cell = po[solps_2d_index(i,j)];
//
//      double r_left = cell_center(leftix[solps_2d_index(i,j)],
//                                  leftiy[solps_2d_index(i,j)],
//                                  crx);
//      double z_left = cell_center(leftix[solps_2d_index(i,j)],
//                                  leftiy[solps_2d_index(i,j)],
//                                  cry);
//      double a_left = po[index_left];
//      double r_right = cell_center(rightix[solps_2d_index(i,j)],
//                                  rightiy[solps_2d_index(i,j)],
//                                  crx);
//      double z_right = cell_center(rightix[solps_2d_index(i,j)],
//                                  rightiy[solps_2d_index(i,j)],
//                                  cry);
//      double a_right = po[index_right];
//      double r_bottom = cell_center(bottomix[solps_2d_index(i,j)],
//                                  bottomiy[solps_2d_index(i,j)],
//                                  crx);
//      double z_bottom = cell_center(bottomix[solps_2d_index(i,j)],
//                                  bottomiy[solps_2d_index(i,j)],
//                                  cry);
//      double a_bottom = po[index_bottom];
//      double r_top = cell_center(topix[solps_2d_index(i,j)],
//                                  topiy[solps_2d_index(i,j)],
//                                  crx);
//      double z_top = cell_center(topix[solps_2d_index(i,j)],
//                                  topiy[solps_2d_index(i,j)],
//                                  cry);
//      double a_top = po[index_top];
//      double r_topright = cell_center(rightix[index_top],
//                                  rightiy[index_top],
//                                  crx);
//      double z_topright = cell_center(rightix[index_top],
//                                      rightiy[index_top],
//                                  cry);
//      double a_topright = po[index_topright];
//      double r_topleft = cell_center(leftix[index_top],
//                                     leftiy[index_top],
//                                  crx);
//      double z_topleft = cell_center(leftix[index_top],
//                                     leftiy[index_top],
//                                  cry);
//      double a_topleft = po[index_topleft];
//      double r_bottomright = cell_center(rightix[index_bottom],
//                                  rightiy[index_bottom],
//                                  crx);
//      double z_bottomright = cell_center(rightix[index_bottom],
//                                      rightiy[index_bottom],
//                                  cry);
//      double a_bottomright = po[index_bottomright];
//      double r_bottomleft = cell_center(leftix[index_bottom],
//                                     leftiy[index_bottom],
//                                  crx);
//      double z_bottomleft = cell_center(leftix[index_bottom],
//                                     leftiy[index_bottom],
//                                  cry);
//      double a_bottomleft = po[index_bottomleft];
//
//            //% Interpolate values at cell edges
//      double d1 = distance(r_cell,z_cell,r_top_mid,z_top_mid);
//      double d2 = distance(r_top_mid,z_top_mid,r_top,z_top);
//      double a_top_mid = (a_top*d1 + a_cell*d2)/(d1+d2);
//      d1 = distance(r_cell,z_cell,r_bottom_mid,z_bottom_mid);
//      d2 = distance(r_bottom_mid,z_bottom_mid,r_bottom,z_bottom);
//      double a_bottom_mid = (a_bottom*d1 + a_cell*d2)/(d1+d2);
//      d1 = distance(r_cell,z_cell,r_right_mid,z_right_mid);
//      d2 = distance(r_right_mid,z_right_mid,r_right,z_right);
//      double a_right_mid = (a_right*d1 + a_cell*d2)/(d1+d2);
//      d1 = distance(r_cell,z_cell,r_left_mid,z_left_mid);
//      d2 = distance(r_left_mid,z_left_mid,r_left,z_left);
//      double a_left_mid = (a_left*d1 + a_cell*d2)/(d1+d2);
//
//      // % Off grid values for corners
//      double r_topright_offgrid = mean(crx[solps_3d_index(i_right,j_right,2)],crx[solps_3d_index(i_right,j_right,3)]);
//      double z_topright_offgrid = mean(cry[solps_3d_index(i_right,j_right,2)],cry[solps_3d_index(i_right,j_right,3)]);
//      d1 = distance(r_right,z_right,r_topright_offgrid,z_topright_offgrid);
//      d2 = distance(r_topright_offgrid,z_topright_offgrid,r_topright,z_topright);
//      double a_topright_offgrid = (a_topright*d1 + a_right*d2)/(d1+d2);
//      d1 = distance(r_top_mid,z_top_mid,r_top_right,z_top_right);
//      d2 = distance(r_top_right,z_top_right,r_topright_offgrid,z_topright_offgrid);
//      double a_topright_corner = (a_topright_offgrid*d1 + a_top_mid*d2)/(d1+d2);
//
//      double r_bottomright_offgrid = mean(crx[solps_3d_index(i_right,j_right,0)],crx[solps_3d_index(i_right,j_right,1)]);
//      double z_bottomright_offgrid = mean(cry[solps_3d_index(i_right,j_right,0)],cry[solps_3d_index(i_right,j_right,1)]);
//      d1 = distance(r_right,z_right,r_bottomright_offgrid,z_bottomright_offgrid);
//      d2 = distance(r_bottomright_offgrid,z_bottomright_offgrid,r_bottomright,z_bottomright);
//      double a_bottomright_offgrid = (a_bottomright*d1 + a_right*d2)/(d1+d2);
//      d1 = distance(r_bottom_mid,z_bottom_mid,r_bottom_right,z_bottom_right);
//      d2 = distance(r_bottom_right,z_bottom_right,r_bottomright_offgrid,z_bottomright_offgrid);
//      double a_bottomright_corner = (a_bottomright_offgrid*d1 + a_bottom_mid*d2)/(d1+d2);
//
//      double r_topleft_offgrid = mean(crx[solps_3d_index(i_left,j_left,2)],crx[solps_3d_index(i_left,j_left,3)]);
//      double z_topleft_offgrid = mean(cry[solps_3d_index(i_left,j_left,2)],cry[solps_3d_index(i_left,j_left,3)]);
//      d1 = distance(r_left,z_left,r_topleft_offgrid,z_topleft_offgrid);
//      d2 = distance(r_topleft_offgrid,z_topleft_offgrid,r_topleft,z_topleft);
//      double a_topleft_offgrid = (a_topleft*d1 + a_left*d2)/(d1+d2);
//      d1 = distance(r_top_mid,z_top_mid,r_top_left,z_top_left);
//      d2 = distance(r_top_left,z_top_left,r_topleft_offgrid,z_topleft_offgrid);
//      double a_topleft_corner = (a_topleft_offgrid*d1 + a_top_mid*d2)/(d1+d2);
//      
//      double r_bottomleft_offgrid = mean(crx[solps_3d_index(i_left,j_left,0)],crx[solps_3d_index(i_left,j_left,1)]);
//      double z_bottomleft_offgrid = mean(cry[solps_3d_index(i_left,j_left,0)],cry[solps_3d_index(i_left,j_left,1)]);
//
//      d1 = distance(r_left,z_left,r_bottomleft_offgrid,z_bottomleft_offgrid);
//      d2 = distance(r_bottomleft_offgrid,z_bottomleft_offgrid,r_bottomleft,z_bottomleft);
//      double a_bottomleft_offgrid = (a_bottomleft*d1 + a_left*d2)/(d1+d2);
//      d1 = distance(r_bottom_mid,z_bottom_mid,r_bottom_left,z_bottom_left);
//      d2 = distance(r_bottom_left,z_bottom_left,r_bottomleft_offgrid,z_bottomleft_offgrid);
//      double a_bottomleft_corner = (a_bottomleft_offgrid*d1 + a_bottom_mid*d2)/(d1+d2);
//      int triangle_index = solps_3d_index_store(i-1,j-1,0);
//      r1[triangle_index] = r_cell;
//      z1[triangle_index] = z_cell;
//      v1[triangle_index] = a_cell;
//      r2[triangle_index] = r_top_mid;
//      z2[triangle_index] = z_top_mid;
//      v2[triangle_index] = a_top_mid;
//      r3[triangle_index] = r_top_right;
//      z3[triangle_index] = z_top_right;
//      v3[triangle_index] = a_topright_corner;
//      double d12 = distance(r1[triangle_index],z1[triangle_index],r2[triangle_index],z2[triangle_index]); 
//      double d23 = distance(r3[triangle_index],z3[triangle_index],r2[triangle_index],z2[triangle_index]); 
//      double d13 = distance(r1[triangle_index],z1[triangle_index],r3[triangle_index],z3[triangle_index]); 
//      double max1 = std::max(d12,d23);
//      radius[triangle_index] = std::max(max1,d13);
//
//      triangle_index = solps_3d_index_store(i-1,j-1,1);
//      r1[triangle_index] = r_cell;
//      z1[triangle_index] = z_cell;
//      v1[triangle_index] = a_cell;
//      r2[triangle_index] = r_top_right;
//      z2[triangle_index] = z_top_right;
//      v2[triangle_index] = a_topright_corner;
//      r3[triangle_index] = r_right_mid;
//      z3[triangle_index] = z_right_mid;
//      v3[triangle_index] = a_right_mid;
//      d12 = distance(r1[triangle_index],z1[triangle_index],r2[triangle_index],z2[triangle_index]); 
//      d23 = distance(r3[triangle_index],z3[triangle_index],r2[triangle_index],z2[triangle_index]); 
//      d13 = distance(r1[triangle_index],z1[triangle_index],r3[triangle_index],z3[triangle_index]); 
//      max1 = std::max(d12,d23);
//      radius[triangle_index] = std::max(max1,d13);
//      
//      triangle_index = solps_3d_index_store(i-1,j-1,2);
//      r1[triangle_index] = r_cell;
//      z1[triangle_index] = z_cell;
//      v1[triangle_index] = a_cell;
//      r2[triangle_index] = r_right_mid;
//      z2[triangle_index] = z_right_mid;
//      v2[triangle_index] = a_right_mid;
//      r3[triangle_index] = r_bottom_right;
//      z3[triangle_index] = z_bottom_right;
//      v3[triangle_index] = a_bottomright_corner;
//      d12 = distance(r1[triangle_index],z1[triangle_index],r2[triangle_index],z2[triangle_index]); 
//      d23 = distance(r3[triangle_index],z3[triangle_index],r2[triangle_index],z2[triangle_index]); 
//      d13 = distance(r1[triangle_index],z1[triangle_index],r3[triangle_index],z3[triangle_index]); 
//      max1 = std::max(d12,d23);
//      radius[triangle_index] = std::max(max1,d13);
//      
//      triangle_index = solps_3d_index_store(i-1,j-1,3);
//      r1[triangle_index] = r_cell;
//      z1[triangle_index] = z_cell;
//      v1[triangle_index] = a_cell;
//      r2[triangle_index] = r_bottom_right;
//      z2[triangle_index] = z_bottom_right;
//      v2[triangle_index] = a_bottomright_corner;
//      r3[triangle_index] = r_bottom_mid;
//      z3[triangle_index] = z_bottom_mid;
//      v3[triangle_index] = a_bottom_mid;
//      d12 = distance(r1[triangle_index],z1[triangle_index],r2[triangle_index],z2[triangle_index]); 
//      d23 = distance(r3[triangle_index],z3[triangle_index],r2[triangle_index],z2[triangle_index]); 
//      d13 = distance(r1[triangle_index],z1[triangle_index],r3[triangle_index],z3[triangle_index]); 
//      max1 = std::max(d12,d23);
//      radius[triangle_index] = std::max(max1,d13);
//      
//      triangle_index = solps_3d_index_store(i-1,j-1,4);
//      r1[triangle_index] = r_cell;
//      z1[triangle_index] = z_cell;
//      v1[triangle_index] = a_cell;
//      r2[triangle_index] = r_bottom_mid;
//      z2[triangle_index] = z_bottom_mid;
//      v2[triangle_index] = a_bottom_mid;
//      r3[triangle_index] = r_bottom_left;
//      z3[triangle_index] = z_bottom_left;
//      v3[triangle_index] = a_bottomleft_corner;
//      d12 = distance(r1[triangle_index],z1[triangle_index],r2[triangle_index],z2[triangle_index]); 
//      d23 = distance(r3[triangle_index],z3[triangle_index],r2[triangle_index],z2[triangle_index]); 
//      d13 = distance(r1[triangle_index],z1[triangle_index],r3[triangle_index],z3[triangle_index]); 
//      max1 = std::max(d12,d23);
//      radius[triangle_index] = std::max(max1,d13);
//      
//      triangle_index = solps_3d_index_store(i-1,j-1,5);
//      r1[triangle_index] = r_cell;
//      z1[triangle_index] = z_cell;
//      v1[triangle_index] = a_cell;
//      r2[triangle_index] = r_bottom_left;
//      z2[triangle_index] = z_bottom_left;
//      v2[triangle_index] = a_bottomleft_corner;
//      r3[triangle_index] = r_left_mid;
//      z3[triangle_index] = z_left_mid;
//      v3[triangle_index] = a_left_mid;
//      d12 = distance(r1[triangle_index],z1[triangle_index],r2[triangle_index],z2[triangle_index]); 
//      d23 = distance(r3[triangle_index],z3[triangle_index],r2[triangle_index],z2[triangle_index]); 
//      d13 = distance(r1[triangle_index],z1[triangle_index],r3[triangle_index],z3[triangle_index]); 
//      max1 = std::max(d12,d23);
//      radius[triangle_index] = std::max(max1,d13);
//      
//      triangle_index = solps_3d_index_store(i-1,j-1,6);
//      r1[triangle_index] = r_cell;
//      z1[triangle_index] = z_cell;
//      v1[triangle_index] = a_cell;
//      r2[triangle_index] = r_left_mid;
//      z2[triangle_index] = z_left_mid;
//      v2[triangle_index] = a_left_mid;
//      r3[triangle_index] = r_top_left;
//      z3[triangle_index] = z_top_left;
//      v3[triangle_index] = a_topleft_corner;
//      d12 = distance(r1[triangle_index],z1[triangle_index],r2[triangle_index],z2[triangle_index]); 
//      d23 = distance(r3[triangle_index],z3[triangle_index],r2[triangle_index],z2[triangle_index]); 
//      d13 = distance(r1[triangle_index],z1[triangle_index],r3[triangle_index],z3[triangle_index]); 
//      max1 = std::max(d12,d23);
//      radius[triangle_index] = std::max(max1,d13);
//      
//      triangle_index = solps_3d_index_store(i-1,j-1,7);
//      r1[triangle_index] = r_cell;
//      z1[triangle_index] = z_cell;
//      v1[triangle_index] = a_cell;
//      r2[triangle_index] = r_top_left;
//      z2[triangle_index] = z_top_left;
//      v2[triangle_index] = a_topleft_corner;
//      r3[triangle_index] = r_top_mid;
//      z3[triangle_index] = z_top_mid;
//      v3[triangle_index] = a_top_mid;
//      d12 = distance(r1[triangle_index],z1[triangle_index],r2[triangle_index],z2[triangle_index]); 
//      d23 = distance(r3[triangle_index],z3[triangle_index],r2[triangle_index],z2[triangle_index]); 
//      d13 = distance(r1[triangle_index],z1[triangle_index],r3[triangle_index],z3[triangle_index]); 
//      max1 = std::max(d12,d23);
//      radius[triangle_index] = std::max(max1,d13);
//      
//      if(i==1)
//      {
//	        double target_buff = 0.01;
//		r1_edge[(j-1)*4 + 0] = r_top_left;
//		z1_edge[(j-1)*4 + 0] = z_top_left;
//		r2_edge[(j-1)*4 + 0] = r_bottom_left;
//		z2_edge[(j-1)*4 + 0] = z_bottom_left;
//		v1_edge[(j-1)*4 + 0] = a_topleft_corner;
//		v2_edge[(j-1)*4 + 0] = a_bottomleft_corner;
//
//		r1_edge[(j-1)*4 + 1] = r_bottom_left;
//		z1_edge[(j-1)*4 + 1] = z_bottom_left;
//		v1_edge[(j-1)*4 + 1] = a_bottomleft_corner;
//
//
//                double topvecr = r_top_left - r_top_right;
//		double topvecz = z_top_left - z_top_right;
//                double topvecnorm = std::sqrt(topvecr*topvecr + topvecz*topvecz);
//
//                double bottomvecr = r_bottom_left - r_bottom_right;
//		double bottomvecz = z_bottom_left - z_bottom_right;
//                double bottomvecnorm = std::sqrt(bottomvecr*bottomvecr + bottomvecz*bottomvecz);
//                
//		r2_edge[(j-1)*4 + 1] = r_top_left + target_buff*topvecr;
//		z2_edge[(j-1)*4 + 1] = z_top_left + target_buff*topvecz;
//		v2_edge[(j-1)*4 + 1] = a_topleft_corner;
//		
//		r3_edge[(j-1)*4 + 0] = r_top_left + target_buff*topvecr;
//		r3_edge[(j-1)*4 + 1] = r_bottom_left + target_buff*bottomvecr;
//		z3_edge[(j-1)*4 + 0] = z_top_left + target_buff*topvecz;
//		z3_edge[(j-1)*4 + 1] = z_bottom_left + target_buff*bottomvecz;
//		v3_edge[(j-1)*4 + 0] = a_topleft_corner;
//		v3_edge[(j-1)*4 + 1] = a_bottomleft_corner;
//      
//		d12 = distance(r1_edge[(j-1)*4 + 0],z1_edge[(j-1)*4 + 0],
//			       r2_edge[(j-1)*4 + 0],z2_edge[(j-1)*4 + 0]); 
//		d23 = distance(r3_edge[(j-1)*4 + 0],z3_edge[(j-1)*4 + 0],
//			       r2_edge[(j-1)*4 + 0],z2_edge[(j-1)*4 + 0]); 
//		d13 = distance(r1_edge[(j-1)*4 + 0],z1_edge[(j-1)*4 + 0],
//			       r3_edge[(j-1)*4 + 0],z3_edge[(j-1)*4 + 0]); 
//      		max1 = std::max(d12,d23);
//      		radius_edge[(j-1)*4 + 0] = std::max(max1,d13);
//		
//		d12 = distance(r1_edge[(j-1)*4 + 1],z1_edge[(j-1)*4 + 1],
//			       r2_edge[(j-1)*4 + 1],z2_edge[(j-1)*4 + 1]); 
//		d23 = distance(r3_edge[(j-1)*4 + 1],z3_edge[(j-1)*4 + 1],
//			       r2_edge[(j-1)*4 + 1],z2_edge[(j-1)*4 + 1]); 
//		d13 = distance(r1_edge[(j-1)*4 + 1],z1_edge[(j-1)*4 + 1],
//			       r3_edge[(j-1)*4 + 1],z3_edge[(j-1)*4 + 1]); 
//      		max1 = std::max(d12,d23);
//      		radius_edge[(j-1)*4 + 1] = std::max(max1,d13);
//      }
//
//    }
//  }
//
//  for(int ii = 0; ii< 2*ny;ii++)
//  {
//	  v1.push_back(v1_edge[ii]);
//	  v2.push_back(v2_edge[ii]);
//	  v3.push_back(v3_edge[ii]);
//  }
//
//  return std::make_tuple(v1, v2, v3);
//}

std::tuple<std::vector<double>,std::vector<double>,std::vector<double>>
            get_scalar_field_tris(std::vector<double> solps_field,
           std::vector<double> v1,std::vector<double> v2,
           std::vector<double> v3)
{
           std::vector<double> r1, r2, r3, z1, z2, z3, radius;
  
           std::vector<int> nxny = read_ifield("b2fgmtry","nx,ny");

  nx = nxny[0];
  ny = nxny[1];

  // Calculate Efield on SOLPS grid
  int n_total = nx*ny*8;
  r1.resize(n_total,0.0);
  r2.resize(n_total,0.0);
  r3.resize(n_total,0.0);
  z1.resize(n_total,0.0);
  z2.resize(n_total,0.0);
  z3.resize(n_total,0.0);
  v1.resize(n_total,0.0);
  v2.resize(n_total,0.0);
  v3.resize(n_total,0.0);
  radius.resize(n_total,0.0);
  
  int n_edge_total = 4*ny;
  std::vector<double> r1_edge(n_edge_total,0.0);
  std::vector<double> r2_edge(n_edge_total,0.0);
  std::vector<double> r3_edge(n_edge_total,0.0);
  std::vector<double> z1_edge(n_edge_total,0.0);
  std::vector<double> z2_edge(n_edge_total,0.0);
  std::vector<double> z3_edge(n_edge_total,0.0);
  std::vector<double> v1_edge(n_edge_total,0.0);
  std::vector<double> v2_edge(n_edge_total,0.0);
  std::vector<double> v3_edge(n_edge_total,0.0);
  std::vector<double> radius_edge(n_edge_total,0.0);
    
  std::vector<double> crx = read_dfield("b2fgmtry", "crx");
  std::vector<double> cry = read_dfield("b2fgmtry", "cry");
  std::vector<double> hx = read_dfield("b2fgmtry", "hx");
  std::vector<double> hy = read_dfield("b2fgmtry", "hy");
  
  std::vector<int> leftix = read_ifield("b2fgmtry", "leftix");
  std::vector<int> leftiy = read_ifield("b2fgmtry", "leftiy");
  std::vector<int> rightix = read_ifield("b2fgmtry", "rightix");
  std::vector<int> rightiy = read_ifield("b2fgmtry", "rightiy");
  std::vector<int> topix = read_ifield("b2fgmtry", "topix");
  std::vector<int> topiy = read_ifield("b2fgmtry", "topiy");
  std::vector<int> bottomix = read_ifield("b2fgmtry", "bottomix");
  std::vector<int> bottomiy = read_ifield("b2fgmtry", "bottomiy");
    
  for (int i = 0; i< leftix.size(); i++)
          {
            leftix[i] = leftix[i] + 1;
            leftiy[i] = leftiy[i] + 1;
            rightix[i] = rightix[i] + 1;
            rightiy[i] = rightiy[i] + 1;
            topix[i] = topix[i] + 1;
            topiy[i] = topiy[i] + 1;
            bottomix[i] = bottomix[i] + 1;
            bottomiy[i] = bottomiy[i] + 1;
          }
  // Get SOLPS state variables
  //std::vector<double> po = read_dfield("b2fstate", scalar_string);

  for (int i=1; i < nx+1; i++)
  {
    for( int j=1; j < ny+1; j++)
    {
      int cell_2d_index = solps_2d_index(i,j);

      double cell_hx = hx[cell_2d_index];
      double cell_hy = hy[cell_2d_index];
      double cell_po = solps_field[cell_2d_index];

      int top_2d_index = solps_2d_index(topix[solps_2d_index(i,j)],
                                        topiy[solps_2d_index(i,j)]);

      double top_hx = hx[top_2d_index];
      double top_hy = hy[top_2d_index];
      double top_po = solps_field[top_2d_index];

      int bottom_2d_index = solps_2d_index(bottomix[solps_2d_index(i,j)],
                                        bottomiy[solps_2d_index(i,j)]);

      double bottom_hx = hx[bottom_2d_index];
      double bottom_hy = hy[bottom_2d_index];
      double bottom_po = solps_field[bottom_2d_index];

      int right_2d_index = solps_2d_index(rightix[solps_2d_index(i,j)],
                                        rightiy[solps_2d_index(i,j)]);

      double right_hx = hx[right_2d_index];
      double right_hy = hy[right_2d_index];
      double right_po = solps_field[right_2d_index];

      int left_2d_index = solps_2d_index(leftix[solps_2d_index(i,j)],
                                        leftiy[solps_2d_index(i,j)]);

      double left_hx = hx[left_2d_index];
      double left_hy = hy[left_2d_index];
      double left_po = solps_field[left_2d_index];
      
      double r_bottom_left = crx[solps_3d_index(i,j,0)];
      double z_bottom_left = cry[solps_3d_index(i,j,0)];
      double r_bottom_right = crx[solps_3d_index(i,j,1)];
      double z_bottom_right = cry[solps_3d_index(i,j,1)];
      double r_top_left = crx[solps_3d_index(i,j,2)];
      double z_top_left = cry[solps_3d_index(i,j,2)];
      double r_top_right = crx[solps_3d_index(i,j,3)];
      double z_top_right = cry[solps_3d_index(i,j,3)];

      double r_right_mid = mean(r_top_right, r_bottom_right);
      double z_right_mid = mean(z_top_right, z_bottom_right);
      double r_left_mid = mean(r_top_left, r_bottom_left);
      double z_left_mid = mean(z_top_left, z_bottom_left);
      double r_top_mid = mean(r_top_left, r_top_right);
      double z_top_mid = mean(z_top_left, z_top_right);
      double r_bottom_mid = mean(r_bottom_left, r_bottom_right);
      double z_bottom_mid = mean(z_bottom_left, z_bottom_right);

      double norm_left_right = std::sqrt((r_right_mid - r_left_mid)*(r_right_mid - r_left_mid) +
            (z_right_mid - z_left_mid)*(z_right_mid - z_left_mid));

      double norm_bottom_top = std::sqrt((r_top_mid - r_bottom_mid)*(r_top_mid - r_bottom_mid) +
            (z_top_mid - z_bottom_mid)*(z_top_mid - z_bottom_mid));

      int i_left = leftix[solps_2d_index(i,j)];
      int j_left = leftiy[solps_2d_index(i,j)];
      int i_right = rightix[solps_2d_index(i,j)];
      int j_right = rightiy[solps_2d_index(i,j)];
      int index_left = solps_2d_index(leftix[solps_2d_index(i,j)],
                                   leftiy[solps_2d_index(i,j)]);
      int index_right = solps_2d_index(rightix[solps_2d_index(i,j)],
                                    rightiy[solps_2d_index(i,j)]);
      int index_bottom = solps_2d_index(bottomix[solps_2d_index(i,j)],
                                     bottomiy[solps_2d_index(i,j)]);
      int index_top = solps_2d_index(topix[solps_2d_index(i,j)],
                                  topiy[solps_2d_index(i,j)]);
       
      int index_topright = solps_2d_index(rightix[index_top],
                                       rightiy[index_top]);
      int index_topleft = solps_2d_index(leftix[index_top],
                                      leftiy[index_top]);
      int index_bottomright = solps_2d_index(rightix[index_bottom],
                                       rightiy[index_bottom]);
      int index_bottomleft = solps_2d_index(leftix[index_bottom],
                                      leftiy[index_bottom]);
      
      double r_cell = cell_center(i,j,crx);
      double z_cell = cell_center(i,j,cry);
      double a_cell = solps_field[solps_2d_index(i,j)];

      double r_left = cell_center(leftix[solps_2d_index(i,j)],
                                  leftiy[solps_2d_index(i,j)],
                                  crx);
      double z_left = cell_center(leftix[solps_2d_index(i,j)],
                                  leftiy[solps_2d_index(i,j)],
                                  cry);
      double a_left = solps_field[index_left];
      double r_right = cell_center(rightix[solps_2d_index(i,j)],
                                  rightiy[solps_2d_index(i,j)],
                                  crx);
      double z_right = cell_center(rightix[solps_2d_index(i,j)],
                                  rightiy[solps_2d_index(i,j)],
                                  cry);
      double a_right = solps_field[index_right];
      double r_bottom = cell_center(bottomix[solps_2d_index(i,j)],
                                  bottomiy[solps_2d_index(i,j)],
                                  crx);
      double z_bottom = cell_center(bottomix[solps_2d_index(i,j)],
                                  bottomiy[solps_2d_index(i,j)],
                                  cry);
      double a_bottom = solps_field[index_bottom];
      double r_top = cell_center(topix[solps_2d_index(i,j)],
                                  topiy[solps_2d_index(i,j)],
                                  crx);
      double z_top = cell_center(topix[solps_2d_index(i,j)],
                                  topiy[solps_2d_index(i,j)],
                                  cry);
      double a_top = solps_field[index_top];
      double r_topright = cell_center(rightix[index_top],
                                  rightiy[index_top],
                                  crx);
      double z_topright = cell_center(rightix[index_top],
                                      rightiy[index_top],
                                  cry);
      double a_topright = solps_field[index_topright];
      double r_topleft = cell_center(leftix[index_top],
                                     leftiy[index_top],
                                  crx);
      double z_topleft = cell_center(leftix[index_top],
                                     leftiy[index_top],
                                  cry);
      double a_topleft = solps_field[index_topleft];
      double r_bottomright = cell_center(rightix[index_bottom],
                                  rightiy[index_bottom],
                                  crx);
      double z_bottomright = cell_center(rightix[index_bottom],
                                      rightiy[index_bottom],
                                  cry);
      double a_bottomright = solps_field[index_bottomright];
      double r_bottomleft = cell_center(leftix[index_bottom],
                                     leftiy[index_bottom],
                                  crx);
      double z_bottomleft = cell_center(leftix[index_bottom],
                                     leftiy[index_bottom],
                                  cry);
      double a_bottomleft = solps_field[index_bottomleft];

            //% Interpolate values at cell edges
      double d1 = distance(r_cell,z_cell,r_top_mid,z_top_mid);
      double d2 = distance(r_top_mid,z_top_mid,r_top,z_top);
      double a_top_mid = (a_top*d1 + a_cell*d2)/(d1+d2);
      d1 = distance(r_cell,z_cell,r_bottom_mid,z_bottom_mid);
      d2 = distance(r_bottom_mid,z_bottom_mid,r_bottom,z_bottom);
      double a_bottom_mid = (a_bottom*d1 + a_cell*d2)/(d1+d2);
      d1 = distance(r_cell,z_cell,r_right_mid,z_right_mid);
      d2 = distance(r_right_mid,z_right_mid,r_right,z_right);
      double a_right_mid = (a_right*d1 + a_cell*d2)/(d1+d2);
      d1 = distance(r_cell,z_cell,r_left_mid,z_left_mid);
      d2 = distance(r_left_mid,z_left_mid,r_left,z_left);
      double a_left_mid = (a_left*d1 + a_cell*d2)/(d1+d2);

      // % Off grid values for corners
      double r_topright_offgrid = mean(crx[solps_3d_index(i_right,j_right,2)],crx[solps_3d_index(i_right,j_right,3)]);
      double z_topright_offgrid = mean(cry[solps_3d_index(i_right,j_right,2)],cry[solps_3d_index(i_right,j_right,3)]);
      d1 = distance(r_right,z_right,r_topright_offgrid,z_topright_offgrid);
      d2 = distance(r_topright_offgrid,z_topright_offgrid,r_topright,z_topright);
      double a_topright_offgrid = (a_topright*d1 + a_right*d2)/(d1+d2);
      d1 = distance(r_top_mid,z_top_mid,r_top_right,z_top_right);
      d2 = distance(r_top_right,z_top_right,r_topright_offgrid,z_topright_offgrid);
      double a_topright_corner = (a_topright_offgrid*d1 + a_top_mid*d2)/(d1+d2);

      double r_bottomright_offgrid = mean(crx[solps_3d_index(i_right,j_right,0)],crx[solps_3d_index(i_right,j_right,1)]);
      double z_bottomright_offgrid = mean(cry[solps_3d_index(i_right,j_right,0)],cry[solps_3d_index(i_right,j_right,1)]);
      d1 = distance(r_right,z_right,r_bottomright_offgrid,z_bottomright_offgrid);
      d2 = distance(r_bottomright_offgrid,z_bottomright_offgrid,r_bottomright,z_bottomright);
      double a_bottomright_offgrid = (a_bottomright*d1 + a_right*d2)/(d1+d2);
      d1 = distance(r_bottom_mid,z_bottom_mid,r_bottom_right,z_bottom_right);
      d2 = distance(r_bottom_right,z_bottom_right,r_bottomright_offgrid,z_bottomright_offgrid);
      double a_bottomright_corner = (a_bottomright_offgrid*d1 + a_bottom_mid*d2)/(d1+d2);

      double r_topleft_offgrid = mean(crx[solps_3d_index(i_left,j_left,2)],crx[solps_3d_index(i_left,j_left,3)]);
      double z_topleft_offgrid = mean(cry[solps_3d_index(i_left,j_left,2)],cry[solps_3d_index(i_left,j_left,3)]);
      d1 = distance(r_left,z_left,r_topleft_offgrid,z_topleft_offgrid);
      d2 = distance(r_topleft_offgrid,z_topleft_offgrid,r_topleft,z_topleft);
      double a_topleft_offgrid = (a_topleft*d1 + a_left*d2)/(d1+d2);
      d1 = distance(r_top_mid,z_top_mid,r_top_left,z_top_left);
      d2 = distance(r_top_left,z_top_left,r_topleft_offgrid,z_topleft_offgrid);
      double a_topleft_corner = (a_topleft_offgrid*d1 + a_top_mid*d2)/(d1+d2);
      
      double r_bottomleft_offgrid = mean(crx[solps_3d_index(i_left,j_left,0)],crx[solps_3d_index(i_left,j_left,1)]);
      double z_bottomleft_offgrid = mean(cry[solps_3d_index(i_left,j_left,0)],cry[solps_3d_index(i_left,j_left,1)]);

      d1 = distance(r_left,z_left,r_bottomleft_offgrid,z_bottomleft_offgrid);
      d2 = distance(r_bottomleft_offgrid,z_bottomleft_offgrid,r_bottomleft,z_bottomleft);
      double a_bottomleft_offgrid = (a_bottomleft*d1 + a_left*d2)/(d1+d2);
      d1 = distance(r_bottom_mid,z_bottom_mid,r_bottom_left,z_bottom_left);
      d2 = distance(r_bottom_left,z_bottom_left,r_bottomleft_offgrid,z_bottomleft_offgrid);
      double a_bottomleft_corner = (a_bottomleft_offgrid*d1 + a_bottom_mid*d2)/(d1+d2);
      int triangle_index = solps_3d_index_store(i-1,j-1,0);
      r1[triangle_index] = r_cell;
      z1[triangle_index] = z_cell;
      v1[triangle_index] = a_cell;
      r2[triangle_index] = r_top_mid;
      z2[triangle_index] = z_top_mid;
      v2[triangle_index] = a_top_mid;
      r3[triangle_index] = r_top_right;
      z3[triangle_index] = z_top_right;
      v3[triangle_index] = a_topright_corner;
      double d12 = distance(r1[triangle_index],z1[triangle_index],r2[triangle_index],z2[triangle_index]); 
      double d23 = distance(r3[triangle_index],z3[triangle_index],r2[triangle_index],z2[triangle_index]); 
      double d13 = distance(r1[triangle_index],z1[triangle_index],r3[triangle_index],z3[triangle_index]); 
      double max1 = std::max(d12,d23);
      radius[triangle_index] = std::max(max1,d13);

      triangle_index = solps_3d_index_store(i-1,j-1,1);
      r1[triangle_index] = r_cell;
      z1[triangle_index] = z_cell;
      v1[triangle_index] = a_cell;
      r2[triangle_index] = r_top_right;
      z2[triangle_index] = z_top_right;
      v2[triangle_index] = a_topright_corner;
      r3[triangle_index] = r_right_mid;
      z3[triangle_index] = z_right_mid;
      v3[triangle_index] = a_right_mid;
      d12 = distance(r1[triangle_index],z1[triangle_index],r2[triangle_index],z2[triangle_index]); 
      d23 = distance(r3[triangle_index],z3[triangle_index],r2[triangle_index],z2[triangle_index]); 
      d13 = distance(r1[triangle_index],z1[triangle_index],r3[triangle_index],z3[triangle_index]); 
      max1 = std::max(d12,d23);
      radius[triangle_index] = std::max(max1,d13);
      
      triangle_index = solps_3d_index_store(i-1,j-1,2);
      r1[triangle_index] = r_cell;
      z1[triangle_index] = z_cell;
      v1[triangle_index] = a_cell;
      r2[triangle_index] = r_right_mid;
      z2[triangle_index] = z_right_mid;
      v2[triangle_index] = a_right_mid;
      r3[triangle_index] = r_bottom_right;
      z3[triangle_index] = z_bottom_right;
      v3[triangle_index] = a_bottomright_corner;
      d12 = distance(r1[triangle_index],z1[triangle_index],r2[triangle_index],z2[triangle_index]); 
      d23 = distance(r3[triangle_index],z3[triangle_index],r2[triangle_index],z2[triangle_index]); 
      d13 = distance(r1[triangle_index],z1[triangle_index],r3[triangle_index],z3[triangle_index]); 
      max1 = std::max(d12,d23);
      radius[triangle_index] = std::max(max1,d13);
      
      triangle_index = solps_3d_index_store(i-1,j-1,3);
      r1[triangle_index] = r_cell;
      z1[triangle_index] = z_cell;
      v1[triangle_index] = a_cell;
      r2[triangle_index] = r_bottom_right;
      z2[triangle_index] = z_bottom_right;
      v2[triangle_index] = a_bottomright_corner;
      r3[triangle_index] = r_bottom_mid;
      z3[triangle_index] = z_bottom_mid;
      v3[triangle_index] = a_bottom_mid;
      d12 = distance(r1[triangle_index],z1[triangle_index],r2[triangle_index],z2[triangle_index]); 
      d23 = distance(r3[triangle_index],z3[triangle_index],r2[triangle_index],z2[triangle_index]); 
      d13 = distance(r1[triangle_index],z1[triangle_index],r3[triangle_index],z3[triangle_index]); 
      max1 = std::max(d12,d23);
      radius[triangle_index] = std::max(max1,d13);
      
      triangle_index = solps_3d_index_store(i-1,j-1,4);
      r1[triangle_index] = r_cell;
      z1[triangle_index] = z_cell;
      v1[triangle_index] = a_cell;
      r2[triangle_index] = r_bottom_mid;
      z2[triangle_index] = z_bottom_mid;
      v2[triangle_index] = a_bottom_mid;
      r3[triangle_index] = r_bottom_left;
      z3[triangle_index] = z_bottom_left;
      v3[triangle_index] = a_bottomleft_corner;
      d12 = distance(r1[triangle_index],z1[triangle_index],r2[triangle_index],z2[triangle_index]); 
      d23 = distance(r3[triangle_index],z3[triangle_index],r2[triangle_index],z2[triangle_index]); 
      d13 = distance(r1[triangle_index],z1[triangle_index],r3[triangle_index],z3[triangle_index]); 
      max1 = std::max(d12,d23);
      radius[triangle_index] = std::max(max1,d13);
      
      triangle_index = solps_3d_index_store(i-1,j-1,5);
      r1[triangle_index] = r_cell;
      z1[triangle_index] = z_cell;
      v1[triangle_index] = a_cell;
      r2[triangle_index] = r_bottom_left;
      z2[triangle_index] = z_bottom_left;
      v2[triangle_index] = a_bottomleft_corner;
      r3[triangle_index] = r_left_mid;
      z3[triangle_index] = z_left_mid;
      v3[triangle_index] = a_left_mid;
      d12 = distance(r1[triangle_index],z1[triangle_index],r2[triangle_index],z2[triangle_index]); 
      d23 = distance(r3[triangle_index],z3[triangle_index],r2[triangle_index],z2[triangle_index]); 
      d13 = distance(r1[triangle_index],z1[triangle_index],r3[triangle_index],z3[triangle_index]); 
      max1 = std::max(d12,d23);
      radius[triangle_index] = std::max(max1,d13);
      
      triangle_index = solps_3d_index_store(i-1,j-1,6);
      r1[triangle_index] = r_cell;
      z1[triangle_index] = z_cell;
      v1[triangle_index] = a_cell;
      r2[triangle_index] = r_left_mid;
      z2[triangle_index] = z_left_mid;
      v2[triangle_index] = a_left_mid;
      r3[triangle_index] = r_top_left;
      z3[triangle_index] = z_top_left;
      v3[triangle_index] = a_topleft_corner;
      d12 = distance(r1[triangle_index],z1[triangle_index],r2[triangle_index],z2[triangle_index]); 
      d23 = distance(r3[triangle_index],z3[triangle_index],r2[triangle_index],z2[triangle_index]); 
      d13 = distance(r1[triangle_index],z1[triangle_index],r3[triangle_index],z3[triangle_index]); 
      max1 = std::max(d12,d23);
      radius[triangle_index] = std::max(max1,d13);
      
      triangle_index = solps_3d_index_store(i-1,j-1,7);
      r1[triangle_index] = r_cell;
      z1[triangle_index] = z_cell;
      v1[triangle_index] = a_cell;
      r2[triangle_index] = r_top_left;
      z2[triangle_index] = z_top_left;
      v2[triangle_index] = a_topleft_corner;
      r3[triangle_index] = r_top_mid;
      z3[triangle_index] = z_top_mid;
      v3[triangle_index] = a_top_mid;
      d12 = distance(r1[triangle_index],z1[triangle_index],r2[triangle_index],z2[triangle_index]); 
      d23 = distance(r3[triangle_index],z3[triangle_index],r2[triangle_index],z2[triangle_index]); 
      d13 = distance(r1[triangle_index],z1[triangle_index],r3[triangle_index],z3[triangle_index]); 
      max1 = std::max(d12,d23);
      radius[triangle_index] = std::max(max1,d13);
      
      if(i==1)
      {
	        double target_buff = 0.01;
		r1_edge[(j-1)*4 + 0] = r_top_left;
		z1_edge[(j-1)*4 + 0] = z_top_left;
		r2_edge[(j-1)*4 + 0] = r_bottom_left;
		z2_edge[(j-1)*4 + 0] = z_bottom_left;
		v1_edge[(j-1)*4 + 0] = a_topleft_corner;
		v2_edge[(j-1)*4 + 0] = a_bottomleft_corner;

		r1_edge[(j-1)*4 + 1] = r_bottom_left;
		z1_edge[(j-1)*4 + 1] = z_bottom_left;
		v1_edge[(j-1)*4 + 1] = a_bottomleft_corner;


                double topvecr = r_top_left - r_top_right;
		double topvecz = z_top_left - z_top_right;
                double topvecnorm = std::sqrt(topvecr*topvecr + topvecz*topvecz);

                double bottomvecr = r_bottom_left - r_bottom_right;
		double bottomvecz = z_bottom_left - z_bottom_right;
                double bottomvecnorm = std::sqrt(bottomvecr*bottomvecr + bottomvecz*bottomvecz);
                
		r2_edge[(j-1)*4 + 1] = r_top_left + target_buff*topvecr;
		z2_edge[(j-1)*4 + 1] = z_top_left + target_buff*topvecz;
		v2_edge[(j-1)*4 + 1] = a_topleft_corner;
		
		r3_edge[(j-1)*4 + 0] = r_top_left + target_buff*topvecr;
		r3_edge[(j-1)*4 + 1] = r_bottom_left + target_buff*bottomvecr;
		z3_edge[(j-1)*4 + 0] = z_top_left + target_buff*topvecz;
		z3_edge[(j-1)*4 + 1] = z_bottom_left + target_buff*bottomvecz;
		v3_edge[(j-1)*4 + 0] = a_topleft_corner;
		v3_edge[(j-1)*4 + 1] = a_bottomleft_corner;
      
		d12 = distance(r1_edge[(j-1)*4 + 0],z1_edge[(j-1)*4 + 0],
			       r2_edge[(j-1)*4 + 0],z2_edge[(j-1)*4 + 0]); 
		d23 = distance(r3_edge[(j-1)*4 + 0],z3_edge[(j-1)*4 + 0],
			       r2_edge[(j-1)*4 + 0],z2_edge[(j-1)*4 + 0]); 
		d13 = distance(r1_edge[(j-1)*4 + 0],z1_edge[(j-1)*4 + 0],
			       r3_edge[(j-1)*4 + 0],z3_edge[(j-1)*4 + 0]); 
      		max1 = std::max(d12,d23);
      		radius_edge[(j-1)*4 + 0] = std::max(max1,d13);
		
		d12 = distance(r1_edge[(j-1)*4 + 1],z1_edge[(j-1)*4 + 1],
			       r2_edge[(j-1)*4 + 1],z2_edge[(j-1)*4 + 1]); 
		d23 = distance(r3_edge[(j-1)*4 + 1],z3_edge[(j-1)*4 + 1],
			       r2_edge[(j-1)*4 + 1],z2_edge[(j-1)*4 + 1]); 
		d13 = distance(r1_edge[(j-1)*4 + 1],z1_edge[(j-1)*4 + 1],
			       r3_edge[(j-1)*4 + 1],z3_edge[(j-1)*4 + 1]); 
      		max1 = std::max(d12,d23);
      		radius_edge[(j-1)*4 + 1] = std::max(max1,d13);
      }
      
      if(i==nx)
      {
	        double target_buff = 0.01;
		r1_edge[(j-1)*4 + 2] = r_top_right;
		z1_edge[(j-1)*4 + 2] = z_top_right;
		r2_edge[(j-1)*4 + 2] = r_bottom_right;
		z2_edge[(j-1)*4 + 2] = z_bottom_right;
		v1_edge[(j-1)*4 + 2] = a_topright_corner;
		v2_edge[(j-1)*4 + 2] = a_bottomright_corner;

		r1_edge[(j-1)*4 + 3] = r_bottom_right;
		z1_edge[(j-1)*4 + 3] = z_bottom_right;
		v1_edge[(j-1)*4 + 3] = a_bottomright_corner;


                double topvecr = -(r_top_left - r_top_right);
		double topvecz = -(z_top_left - z_top_right);
                double topvecnorm = std::sqrt(topvecr*topvecr + topvecz*topvecz);
		topvecr = topvecr/topvecnorm;
		topvecz = topvecz/topvecnorm;

                double bottomvecr = -(r_bottom_left - r_bottom_right);
		double bottomvecz = -(z_bottom_left - z_bottom_right);
                double bottomvecnorm = std::sqrt(bottomvecr*bottomvecr + bottomvecz*bottomvecz);
		bottomvecr = bottomvecr/bottomvecnorm;
		bottomvecz = bottomvecz/bottomvecnorm;
                
		r2_edge[(j-1)*4 + 3] = r_top_right + target_buff*topvecr;
		z2_edge[(j-1)*4 + 3] = z_top_right + target_buff*topvecz;
		v2_edge[(j-1)*4 + 3] = a_topright_corner;
		
		r3_edge[(j-1)*4 + 2] = r_top_right + target_buff*topvecr;
		r3_edge[(j-1)*4 + 3] = r_bottom_right + target_buff*bottomvecr;
		z3_edge[(j-1)*4 + 2] = z_top_right + target_buff*topvecz;
		z3_edge[(j-1)*4 + 3] = z_bottom_right + target_buff*bottomvecz;
		v3_edge[(j-1)*4 + 2] = a_topright_corner;
		v3_edge[(j-1)*4 + 3] = a_bottomright_corner;
      
		d12 = distance(r1_edge[(j-1)*4 + 2],z1_edge[(j-1)*4 + 2],
			       r2_edge[(j-1)*4 + 2],z2_edge[(j-1)*4 + 2]); 
		d23 = distance(r3_edge[(j-1)*4 + 2],z3_edge[(j-1)*4 + 2],
			       r2_edge[(j-1)*4 + 2],z2_edge[(j-1)*4 + 2]); 
		d13 = distance(r1_edge[(j-1)*4 + 2],z1_edge[(j-1)*4 + 2],
			       r3_edge[(j-1)*4 + 2],z3_edge[(j-1)*4 + 2]); 
      		max1 = std::max(d12,d23);
      		radius_edge[(j-1)*4 + 2] = std::max(max1,d13);
		
		d12 = distance(r1_edge[(j-1)*4 + 3],z1_edge[(j-1)*4 + 3],
			       r2_edge[(j-1)*4 + 3],z2_edge[(j-1)*4 + 3]); 
		d23 = distance(r3_edge[(j-1)*4 + 3],z3_edge[(j-1)*4 + 3],
			       r2_edge[(j-1)*4 + 3],z2_edge[(j-1)*4 + 3]); 
		d13 = distance(r1_edge[(j-1)*4 + 3],z1_edge[(j-1)*4 + 3],
			       r3_edge[(j-1)*4 + 3],z3_edge[(j-1)*4 + 3]); 
      		max1 = std::max(d12,d23);
      		radius_edge[(j-1)*4 + 3] = std::max(max1,d13);
      }

    }
  }

  for(int ii = 0; ii< 4*ny;ii++)
  {
	  v1.push_back(v1_edge[ii]);
	  v2.push_back(v2_edge[ii]);
	  v3.push_back(v3_edge[ii]);
  }
    

  return std::make_tuple(v1, v2, v3);
}

//std::tuple<std::vector<double>,std::vector<double>,std::vector<double>>
//            get_scalar_field_tris_left(std::vector<double> solps_field,
//           std::vector<double> v1,std::vector<double> v2,
//           std::vector<double> v3)
//{
//           std::vector<double> r1, r2, r3, z1, z2, z3, radius;
//  
//           std::vector<int> nxny = read_ifield("b2fgmtry","nx,ny");
//
//  nx = nxny[0];
//  ny = nxny[1];
//
//  // Calculate Efield on SOLPS grid
//  int n_total = nx*ny*8;
//  r1.resize(n_total,0.0);
//  r2.resize(n_total,0.0);
//  r3.resize(n_total,0.0);
//  z1.resize(n_total,0.0);
//  z2.resize(n_total,0.0);
//  z3.resize(n_total,0.0);
//  v1.resize(n_total,0.0);
//  v2.resize(n_total,0.0);
//  v3.resize(n_total,0.0);
//  radius.resize(n_total,0.0);
//  
//  int n_edge_total = 4*ny;
//  std::vector<double> r1_edge(n_edge_total,0.0);
//  std::vector<double> r2_edge(n_edge_total,0.0);
//  std::vector<double> r3_edge(n_edge_total,0.0);
//  std::vector<double> z1_edge(n_edge_total,0.0);
//  std::vector<double> z2_edge(n_edge_total,0.0);
//  std::vector<double> z3_edge(n_edge_total,0.0);
//  std::vector<double> v1_edge(n_edge_total,0.0);
//  std::vector<double> v2_edge(n_edge_total,0.0);
//  std::vector<double> v3_edge(n_edge_total,0.0);
//  std::vector<double> radius_edge(n_edge_total,0.0);
//    
//  std::vector<double> crx = read_dfield("b2fgmtry", "crx");
//  std::vector<double> cry = read_dfield("b2fgmtry", "cry");
//  std::vector<double> hx = read_dfield("b2fgmtry", "hx");
//  std::vector<double> hy = read_dfield("b2fgmtry", "hy");
//  
//  std::vector<int> leftix = read_ifield("b2fgmtry", "leftix");
//  std::vector<int> leftiy = read_ifield("b2fgmtry", "leftiy");
//  std::vector<int> rightix = read_ifield("b2fgmtry", "rightix");
//  std::vector<int> rightiy = read_ifield("b2fgmtry", "rightiy");
//  std::vector<int> topix = read_ifield("b2fgmtry", "topix");
//  std::vector<int> topiy = read_ifield("b2fgmtry", "topiy");
//  std::vector<int> bottomix = read_ifield("b2fgmtry", "bottomix");
//  std::vector<int> bottomiy = read_ifield("b2fgmtry", "bottomiy");
//    
//  for (int i = 0; i< leftix.size(); i++)
//          {
//            leftix[i] = leftix[i] + 1;
//            leftiy[i] = leftiy[i] + 1;
//            rightix[i] = rightix[i] + 1;
//            rightiy[i] = rightiy[i] + 1;
//            topix[i] = topix[i] + 1;
//            topiy[i] = topiy[i] + 1;
//            bottomix[i] = bottomix[i] + 1;
//            bottomiy[i] = bottomiy[i] + 1;
//          }
//  // Get SOLPS state variables
//  //std::vector<double> po = read_dfield("b2fstate", scalar_string);
//
//  for (int i=1; i < nx+1; i++)
//  {
//    for( int j=1; j < ny+1; j++)
//    {
//      int cell_2d_index = solps_2d_index(i,j);
//
//      double cell_hx = hx[cell_2d_index];
//      double cell_hy = hy[cell_2d_index];
//      double cell_po = solps_field[cell_2d_index];
//
//      int top_2d_index = solps_2d_index(topix[solps_2d_index(i,j)],
//                                        topiy[solps_2d_index(i,j)]);
//
//      double top_hx = hx[top_2d_index];
//      double top_hy = hy[top_2d_index];
//      double top_po = solps_field[top_2d_index];
//
//      int bottom_2d_index = solps_2d_index(bottomix[solps_2d_index(i,j)],
//                                        bottomiy[solps_2d_index(i,j)]);
//
//      double bottom_hx = hx[bottom_2d_index];
//      double bottom_hy = hy[bottom_2d_index];
//      double bottom_po = solps_field[bottom_2d_index];
//
//      int right_2d_index = solps_2d_index(rightix[solps_2d_index(i,j)],
//                                        rightiy[solps_2d_index(i,j)]);
//
//      double right_hx = hx[right_2d_index];
//      double right_hy = hy[right_2d_index];
//      double right_po = solps_field[right_2d_index];
//
//      int left_2d_index = solps_2d_index(leftix[solps_2d_index(i,j)],
//                                        leftiy[solps_2d_index(i,j)]);
//
//      double left_hx = hx[left_2d_index];
//      double left_hy = hy[left_2d_index];
//      double left_po = solps_field[left_2d_index];
//      
//      double r_bottom_left = crx[solps_3d_index(i,j,0)];
//      double z_bottom_left = cry[solps_3d_index(i,j,0)];
//      double r_bottom_right = crx[solps_3d_index(i,j,1)];
//      double z_bottom_right = cry[solps_3d_index(i,j,1)];
//      double r_top_left = crx[solps_3d_index(i,j,2)];
//      double z_top_left = cry[solps_3d_index(i,j,2)];
//      double r_top_right = crx[solps_3d_index(i,j,3)];
//      double z_top_right = cry[solps_3d_index(i,j,3)];
//
//      double r_right_mid = mean(r_top_right, r_bottom_right);
//      double z_right_mid = mean(z_top_right, z_bottom_right);
//      double r_left_mid = mean(r_top_left, r_bottom_left);
//      double z_left_mid = mean(z_top_left, z_bottom_left);
//      double r_top_mid = mean(r_top_left, r_top_right);
//      double z_top_mid = mean(z_top_left, z_top_right);
//      double r_bottom_mid = mean(r_bottom_left, r_bottom_right);
//      double z_bottom_mid = mean(z_bottom_left, z_bottom_right);
//
//      double norm_left_right = std::sqrt((r_right_mid - r_left_mid)*(r_right_mid - r_left_mid) +
//            (z_right_mid - z_left_mid)*(z_right_mid - z_left_mid));
//
//      double norm_bottom_top = std::sqrt((r_top_mid - r_bottom_mid)*(r_top_mid - r_bottom_mid) +
//            (z_top_mid - z_bottom_mid)*(z_top_mid - z_bottom_mid));
//
//      int i_left = leftix[solps_2d_index(i,j)];
//      int j_left = leftiy[solps_2d_index(i,j)];
//      int i_right = rightix[solps_2d_index(i,j)];
//      int j_right = rightiy[solps_2d_index(i,j)];
//      int index_left = solps_2d_index(leftix[solps_2d_index(i,j)],
//                                   leftiy[solps_2d_index(i,j)]);
//      int index_right = solps_2d_index(rightix[solps_2d_index(i,j)],
//                                    rightiy[solps_2d_index(i,j)]);
//      int index_bottom = solps_2d_index(bottomix[solps_2d_index(i,j)],
//                                     bottomiy[solps_2d_index(i,j)]);
//      int index_top = solps_2d_index(topix[solps_2d_index(i,j)],
//                                  topiy[solps_2d_index(i,j)]);
//       
//      int index_topright = solps_2d_index(rightix[index_top],
//                                       rightiy[index_top]);
//      int index_topleft = solps_2d_index(leftix[index_top],
//                                      leftiy[index_top]);
//      int index_bottomright = solps_2d_index(rightix[index_bottom],
//                                       rightiy[index_bottom]);
//      int index_bottomleft = solps_2d_index(leftix[index_bottom],
//                                      leftiy[index_bottom]);
//      
//      double r_cell = cell_center(i,j,crx);
//      double z_cell = cell_center(i,j,cry);
//      double a_cell = solps_field[solps_2d_index(i,j)];
//
//      double r_left = cell_center(leftix[solps_2d_index(i,j)],
//                                  leftiy[solps_2d_index(i,j)],
//                                  crx);
//      double z_left = cell_center(leftix[solps_2d_index(i,j)],
//                                  leftiy[solps_2d_index(i,j)],
//                                  cry);
//      double a_left = a_cell; //solps_field[index_left];
//      double r_right = cell_center(rightix[solps_2d_index(i,j)],
//                                  rightiy[solps_2d_index(i,j)],
//                                  crx);
//      double z_right = cell_center(rightix[solps_2d_index(i,j)],
//                                  rightiy[solps_2d_index(i,j)],
//                                  cry);
//      double a_right = solps_field[index_right];
//      double r_bottom = cell_center(bottomix[solps_2d_index(i,j)],
//                                  bottomiy[solps_2d_index(i,j)],
//                                  crx);
//      double z_bottom = cell_center(bottomix[solps_2d_index(i,j)],
//                                  bottomiy[solps_2d_index(i,j)],
//                                  cry);
//      double a_bottom = solps_field[index_bottom];
//      double r_top = cell_center(topix[solps_2d_index(i,j)],
//                                  topiy[solps_2d_index(i,j)],
//                                  crx);
//      double z_top = cell_center(topix[solps_2d_index(i,j)],
//                                  topiy[solps_2d_index(i,j)],
//                                  cry);
//      double a_top = solps_field[index_top];
//      double r_topright = cell_center(rightix[index_top],
//                                  rightiy[index_top],
//                                  crx);
//      double z_topright = cell_center(rightix[index_top],
//                                      rightiy[index_top],
//                                  cry);
//      double a_topright = solps_field[index_topright];
//      double r_topleft = cell_center(leftix[index_top],
//                                     leftiy[index_top],
//                                  crx);
//      double z_topleft = cell_center(leftix[index_top],
//                                     leftiy[index_top],
//                                  cry);
//      double a_topleft = a_top; //solps_field[index_topleft];
//      double r_bottomright = cell_center(rightix[index_bottom],
//                                  rightiy[index_bottom],
//                                  crx);
//      double z_bottomright = cell_center(rightix[index_bottom],
//                                      rightiy[index_bottom],
//                                  cry);
//      double a_bottomright = solps_field[index_bottomright];
//      double r_bottomleft = cell_center(leftix[index_bottom],
//                                     leftiy[index_bottom],
//                                  crx);
//      double z_bottomleft = cell_center(leftix[index_bottom],
//                                     leftiy[index_bottom],
//                                  cry);
//      double a_bottomleft = a_bottom; //solps_field[index_bottomleft];
//
//            //% Interpolate values at cell edges
//      double d1 = distance(r_cell,z_cell,r_top_mid,z_top_mid);
//      double d2 = distance(r_top_mid,z_top_mid,r_top,z_top);
//      double a_top_mid = (a_top*d1 + a_cell*d2)/(d1+d2);
//      d1 = distance(r_cell,z_cell,r_bottom_mid,z_bottom_mid);
//      d2 = distance(r_bottom_mid,z_bottom_mid,r_bottom,z_bottom);
//      double a_bottom_mid = (a_bottom*d1 + a_cell*d2)/(d1+d2);
//      d1 = distance(r_cell,z_cell,r_right_mid,z_right_mid);
//      d2 = distance(r_right_mid,z_right_mid,r_right,z_right);
//      double a_right_mid = (a_right*d1 + a_cell*d2)/(d1+d2);
//      d1 = distance(r_cell,z_cell,r_left_mid,z_left_mid);
//      d2 = distance(r_left_mid,z_left_mid,r_left,z_left);
//      double a_left_mid = (a_left*d1 + a_cell*d2)/(d1+d2);
//
//      // % Off grid values for corners
//      double r_topright_offgrid = mean(crx[solps_3d_index(i_right,j_right,2)],crx[solps_3d_index(i_right,j_right,3)]);
//      double z_topright_offgrid = mean(cry[solps_3d_index(i_right,j_right,2)],cry[solps_3d_index(i_right,j_right,3)]);
//      d1 = distance(r_right,z_right,r_topright_offgrid,z_topright_offgrid);
//      d2 = distance(r_topright_offgrid,z_topright_offgrid,r_topright,z_topright);
//      double a_topright_offgrid = (a_topright*d1 + a_right*d2)/(d1+d2);
//      d1 = distance(r_top_mid,z_top_mid,r_top_right,z_top_right);
//      d2 = distance(r_top_right,z_top_right,r_topright_offgrid,z_topright_offgrid);
//      double a_topright_corner = (a_topright_offgrid*d1 + a_top_mid*d2)/(d1+d2);
//
//      double r_bottomright_offgrid = mean(crx[solps_3d_index(i_right,j_right,0)],crx[solps_3d_index(i_right,j_right,1)]);
//      double z_bottomright_offgrid = mean(cry[solps_3d_index(i_right,j_right,0)],cry[solps_3d_index(i_right,j_right,1)]);
//      d1 = distance(r_right,z_right,r_bottomright_offgrid,z_bottomright_offgrid);
//      d2 = distance(r_bottomright_offgrid,z_bottomright_offgrid,r_bottomright,z_bottomright);
//      double a_bottomright_offgrid = (a_bottomright*d1 + a_right*d2)/(d1+d2);
//      d1 = distance(r_bottom_mid,z_bottom_mid,r_bottom_right,z_bottom_right);
//      d2 = distance(r_bottom_right,z_bottom_right,r_bottomright_offgrid,z_bottomright_offgrid);
//      double a_bottomright_corner = (a_bottomright_offgrid*d1 + a_bottom_mid*d2)/(d1+d2);
//
//      double r_topleft_offgrid = mean(crx[solps_3d_index(i_left,j_left,2)],crx[solps_3d_index(i_left,j_left,3)]);
//      double z_topleft_offgrid = mean(cry[solps_3d_index(i_left,j_left,2)],cry[solps_3d_index(i_left,j_left,3)]);
//      d1 = distance(r_left,z_left,r_topleft_offgrid,z_topleft_offgrid);
//      d2 = distance(r_topleft_offgrid,z_topleft_offgrid,r_topleft,z_topleft);
//      double a_topleft_offgrid = (a_topleft*d1 + a_left*d2)/(d1+d2);
//      d1 = distance(r_top_mid,z_top_mid,r_top_left,z_top_left);
//      d2 = distance(r_top_left,z_top_left,r_topleft_offgrid,z_topleft_offgrid);
//      double a_topleft_corner = (a_topleft_offgrid*d1 + a_top_mid*d2)/(d1+d2);
//      
//      double r_bottomleft_offgrid = mean(crx[solps_3d_index(i_left,j_left,0)],crx[solps_3d_index(i_left,j_left,1)]);
//      double z_bottomleft_offgrid = mean(cry[solps_3d_index(i_left,j_left,0)],cry[solps_3d_index(i_left,j_left,1)]);
//
//      d1 = distance(r_left,z_left,r_bottomleft_offgrid,z_bottomleft_offgrid);
//      d2 = distance(r_bottomleft_offgrid,z_bottomleft_offgrid,r_bottomleft,z_bottomleft);
//      double a_bottomleft_offgrid = (a_bottomleft*d1 + a_left*d2)/(d1+d2);
//      d1 = distance(r_bottom_mid,z_bottom_mid,r_bottom_left,z_bottom_left);
//      d2 = distance(r_bottom_left,z_bottom_left,r_bottomleft_offgrid,z_bottomleft_offgrid);
//      double a_bottomleft_corner = (a_bottomleft_offgrid*d1 + a_bottom_mid*d2)/(d1+d2);
//      int triangle_index = solps_3d_index_store(i-1,j-1,0);
//      r1[triangle_index] = r_cell;
//      z1[triangle_index] = z_cell;
//      v1[triangle_index] = a_cell;
//      r2[triangle_index] = r_top_mid;
//      z2[triangle_index] = z_top_mid;
//      v2[triangle_index] = a_top_mid;
//      r3[triangle_index] = r_top_right;
//      z3[triangle_index] = z_top_right;
//      v3[triangle_index] = a_topright_corner;
//      double d12 = distance(r1[triangle_index],z1[triangle_index],r2[triangle_index],z2[triangle_index]); 
//      double d23 = distance(r3[triangle_index],z3[triangle_index],r2[triangle_index],z2[triangle_index]); 
//      double d13 = distance(r1[triangle_index],z1[triangle_index],r3[triangle_index],z3[triangle_index]); 
//      double max1 = std::max(d12,d23);
//      radius[triangle_index] = std::max(max1,d13);
//
//      triangle_index = solps_3d_index_store(i-1,j-1,1);
//      r1[triangle_index] = r_cell;
//      z1[triangle_index] = z_cell;
//      v1[triangle_index] = a_cell;
//      r2[triangle_index] = r_top_right;
//      z2[triangle_index] = z_top_right;
//      v2[triangle_index] = a_topright_corner;
//      r3[triangle_index] = r_right_mid;
//      z3[triangle_index] = z_right_mid;
//      v3[triangle_index] = a_right_mid;
//      d12 = distance(r1[triangle_index],z1[triangle_index],r2[triangle_index],z2[triangle_index]); 
//      d23 = distance(r3[triangle_index],z3[triangle_index],r2[triangle_index],z2[triangle_index]); 
//      d13 = distance(r1[triangle_index],z1[triangle_index],r3[triangle_index],z3[triangle_index]); 
//      max1 = std::max(d12,d23);
//      radius[triangle_index] = std::max(max1,d13);
//      
//      triangle_index = solps_3d_index_store(i-1,j-1,2);
//      r1[triangle_index] = r_cell;
//      z1[triangle_index] = z_cell;
//      v1[triangle_index] = a_cell;
//      r2[triangle_index] = r_right_mid;
//      z2[triangle_index] = z_right_mid;
//      v2[triangle_index] = a_right_mid;
//      r3[triangle_index] = r_bottom_right;
//      z3[triangle_index] = z_bottom_right;
//      v3[triangle_index] = a_bottomright_corner;
//      d12 = distance(r1[triangle_index],z1[triangle_index],r2[triangle_index],z2[triangle_index]); 
//      d23 = distance(r3[triangle_index],z3[triangle_index],r2[triangle_index],z2[triangle_index]); 
//      d13 = distance(r1[triangle_index],z1[triangle_index],r3[triangle_index],z3[triangle_index]); 
//      max1 = std::max(d12,d23);
//      radius[triangle_index] = std::max(max1,d13);
//      
//      triangle_index = solps_3d_index_store(i-1,j-1,3);
//      r1[triangle_index] = r_cell;
//      z1[triangle_index] = z_cell;
//      v1[triangle_index] = a_cell;
//      r2[triangle_index] = r_bottom_right;
//      z2[triangle_index] = z_bottom_right;
//      v2[triangle_index] = a_bottomright_corner;
//      r3[triangle_index] = r_bottom_mid;
//      z3[triangle_index] = z_bottom_mid;
//      v3[triangle_index] = a_bottom_mid;
//      d12 = distance(r1[triangle_index],z1[triangle_index],r2[triangle_index],z2[triangle_index]); 
//      d23 = distance(r3[triangle_index],z3[triangle_index],r2[triangle_index],z2[triangle_index]); 
//      d13 = distance(r1[triangle_index],z1[triangle_index],r3[triangle_index],z3[triangle_index]); 
//      max1 = std::max(d12,d23);
//      radius[triangle_index] = std::max(max1,d13);
//      
//      triangle_index = solps_3d_index_store(i-1,j-1,4);
//      r1[triangle_index] = r_cell;
//      z1[triangle_index] = z_cell;
//      v1[triangle_index] = a_cell;
//      r2[triangle_index] = r_bottom_mid;
//      z2[triangle_index] = z_bottom_mid;
//      v2[triangle_index] = a_bottom_mid;
//      r3[triangle_index] = r_bottom_left;
//      z3[triangle_index] = z_bottom_left;
//      v3[triangle_index] = a_bottomleft_corner;
//      d12 = distance(r1[triangle_index],z1[triangle_index],r2[triangle_index],z2[triangle_index]); 
//      d23 = distance(r3[triangle_index],z3[triangle_index],r2[triangle_index],z2[triangle_index]); 
//      d13 = distance(r1[triangle_index],z1[triangle_index],r3[triangle_index],z3[triangle_index]); 
//      max1 = std::max(d12,d23);
//      radius[triangle_index] = std::max(max1,d13);
//      
//      triangle_index = solps_3d_index_store(i-1,j-1,5);
//      r1[triangle_index] = r_cell;
//      z1[triangle_index] = z_cell;
//      v1[triangle_index] = a_cell;
//      r2[triangle_index] = r_bottom_left;
//      z2[triangle_index] = z_bottom_left;
//      v2[triangle_index] = a_bottomleft_corner;
//      r3[triangle_index] = r_left_mid;
//      z3[triangle_index] = z_left_mid;
//      v3[triangle_index] = a_left_mid;
//      d12 = distance(r1[triangle_index],z1[triangle_index],r2[triangle_index],z2[triangle_index]); 
//      d23 = distance(r3[triangle_index],z3[triangle_index],r2[triangle_index],z2[triangle_index]); 
//      d13 = distance(r1[triangle_index],z1[triangle_index],r3[triangle_index],z3[triangle_index]); 
//      max1 = std::max(d12,d23);
//      radius[triangle_index] = std::max(max1,d13);
//      
//      triangle_index = solps_3d_index_store(i-1,j-1,6);
//      r1[triangle_index] = r_cell;
//      z1[triangle_index] = z_cell;
//      v1[triangle_index] = a_cell;
//      r2[triangle_index] = r_left_mid;
//      z2[triangle_index] = z_left_mid;
//      v2[triangle_index] = a_left_mid;
//      r3[triangle_index] = r_top_left;
//      z3[triangle_index] = z_top_left;
//      v3[triangle_index] = a_topleft_corner;
//      d12 = distance(r1[triangle_index],z1[triangle_index],r2[triangle_index],z2[triangle_index]); 
//      d23 = distance(r3[triangle_index],z3[triangle_index],r2[triangle_index],z2[triangle_index]); 
//      d13 = distance(r1[triangle_index],z1[triangle_index],r3[triangle_index],z3[triangle_index]); 
//      max1 = std::max(d12,d23);
//      radius[triangle_index] = std::max(max1,d13);
//      
//      triangle_index = solps_3d_index_store(i-1,j-1,7);
//      r1[triangle_index] = r_cell;
//      z1[triangle_index] = z_cell;
//      v1[triangle_index] = a_cell;
//      r2[triangle_index] = r_top_left;
//      z2[triangle_index] = z_top_left;
//      v2[triangle_index] = a_topleft_corner;
//      r3[triangle_index] = r_top_mid;
//      z3[triangle_index] = z_top_mid;
//      v3[triangle_index] = a_top_mid;
//      d12 = distance(r1[triangle_index],z1[triangle_index],r2[triangle_index],z2[triangle_index]); 
//      d23 = distance(r3[triangle_index],z3[triangle_index],r2[triangle_index],z2[triangle_index]); 
//      d13 = distance(r1[triangle_index],z1[triangle_index],r3[triangle_index],z3[triangle_index]); 
//      max1 = std::max(d12,d23);
//      radius[triangle_index] = std::max(max1,d13);
//      
//      if(i==1)
//      {
//	        double target_buff = 0.01;
//		r1_edge[(j-1)*4 + 0] = r_top_left;
//		z1_edge[(j-1)*4 + 0] = z_top_left;
//		r2_edge[(j-1)*4 + 0] = r_bottom_left;
//		z2_edge[(j-1)*4 + 0] = z_bottom_left;
//		v1_edge[(j-1)*4 + 0] = a_topleft_corner;
//		v2_edge[(j-1)*4 + 0] = a_bottomleft_corner;
//
//		r1_edge[(j-1)*4 + 1] = r_bottom_left;
//		z1_edge[(j-1)*4 + 1] = z_bottom_left;
//		v1_edge[(j-1)*4 + 1] = a_bottomleft_corner;
//
//
//                double topvecr = r_top_left - r_top_right;
//		double topvecz = z_top_left - z_top_right;
//                double topvecnorm = std::sqrt(topvecr*topvecr + topvecz*topvecz);
//
//                double bottomvecr = r_bottom_left - r_bottom_right;
//		double bottomvecz = z_bottom_left - z_bottom_right;
//                double bottomvecnorm = std::sqrt(bottomvecr*bottomvecr + bottomvecz*bottomvecz);
//                
//		r2_edge[(j-1)*4 + 1] = r_top_left + target_buff*topvecr;
//		z2_edge[(j-1)*4 + 1] = z_top_left + target_buff*topvecz;
//		v2_edge[(j-1)*4 + 1] = a_topleft_corner;
//		
//		r3_edge[(j-1)*4 + 0] = r_top_left + target_buff*topvecr;
//		r3_edge[(j-1)*4 + 1] = r_bottom_left + target_buff*bottomvecr;
//		z3_edge[(j-1)*4 + 0] = z_top_left + target_buff*topvecz;
//		z3_edge[(j-1)*4 + 1] = z_bottom_left + target_buff*bottomvecz;
//		v3_edge[(j-1)*4 + 0] = a_topleft_corner;
//		v3_edge[(j-1)*4 + 1] = a_bottomleft_corner;
//      
//		d12 = distance(r1_edge[(j-1)*4 + 0],z1_edge[(j-1)*4 + 0],
//			       r2_edge[(j-1)*4 + 0],z2_edge[(j-1)*4 + 0]); 
//		d23 = distance(r3_edge[(j-1)*4 + 0],z3_edge[(j-1)*4 + 0],
//			       r2_edge[(j-1)*4 + 0],z2_edge[(j-1)*4 + 0]); 
//		d13 = distance(r1_edge[(j-1)*4 + 0],z1_edge[(j-1)*4 + 0],
//			       r3_edge[(j-1)*4 + 0],z3_edge[(j-1)*4 + 0]); 
//      		max1 = std::max(d12,d23);
//      		radius_edge[(j-1)*4 + 0] = std::max(max1,d13);
//		
//		d12 = distance(r1_edge[(j-1)*4 + 1],z1_edge[(j-1)*4 + 1],
//			       r2_edge[(j-1)*4 + 1],z2_edge[(j-1)*4 + 1]); 
//		d23 = distance(r3_edge[(j-1)*4 + 1],z3_edge[(j-1)*4 + 1],
//			       r2_edge[(j-1)*4 + 1],z2_edge[(j-1)*4 + 1]); 
//		d13 = distance(r1_edge[(j-1)*4 + 1],z1_edge[(j-1)*4 + 1],
//			       r3_edge[(j-1)*4 + 1],z3_edge[(j-1)*4 + 1]); 
//      		max1 = std::max(d12,d23);
//      		radius_edge[(j-1)*4 + 1] = std::max(max1,d13);
//      }
//      
//      if(i==nx)
//      {
//	        double target_buff = 0.01;
//		r1_edge[(j-1)*4 + 2] = r_top_right;
//		z1_edge[(j-1)*4 + 2] = z_top_right;
//		r2_edge[(j-1)*4 + 2] = r_bottom_right;
//		z2_edge[(j-1)*4 + 2] = z_bottom_right;
//		v1_edge[(j-1)*4 + 2] = a_topright_corner;
//		v2_edge[(j-1)*4 + 2] = a_bottomright_corner;
//
//		r1_edge[(j-1)*4 + 3] = r_bottom_right;
//		z1_edge[(j-1)*4 + 3] = z_bottom_right;
//		v1_edge[(j-1)*4 + 3] = a_bottomright_corner;
//
//
//                double topvecr = -(r_top_left - r_top_right);
//		double topvecz = -(z_top_left - z_top_right);
//                double topvecnorm = std::sqrt(topvecr*topvecr + topvecz*topvecz);
//		topvecr = topvecr/topvecnorm;
//		topvecz = topvecz/topvecnorm;
//
//                double bottomvecr = -(r_bottom_left - r_bottom_right);
//		double bottomvecz = -(z_bottom_left - z_bottom_right);
//                double bottomvecnorm = std::sqrt(bottomvecr*bottomvecr + bottomvecz*bottomvecz);
//		bottomvecr = bottomvecr/bottomvecnorm;
//		bottomvecz = bottomvecz/bottomvecnorm;
//                
//		r2_edge[(j-1)*4 + 3] = r_top_right + target_buff*topvecr;
//		z2_edge[(j-1)*4 + 3] = z_top_right + target_buff*topvecz;
//		v2_edge[(j-1)*4 + 3] = a_topright_corner;
//		
//		r3_edge[(j-1)*4 + 2] = r_top_right + target_buff*topvecr;
//		r3_edge[(j-1)*4 + 3] = r_bottom_right + target_buff*bottomvecr;
//		z3_edge[(j-1)*4 + 2] = z_top_right + target_buff*topvecz;
//		z3_edge[(j-1)*4 + 3] = z_bottom_right + target_buff*bottomvecz;
//		v3_edge[(j-1)*4 + 2] = a_topright_corner;
//		v3_edge[(j-1)*4 + 3] = a_bottomright_corner;
//      
//		d12 = distance(r1_edge[(j-1)*4 + 2],z1_edge[(j-1)*4 + 2],
//			       r2_edge[(j-1)*4 + 2],z2_edge[(j-1)*4 + 2]); 
//		d23 = distance(r3_edge[(j-1)*4 + 2],z3_edge[(j-1)*4 + 2],
//			       r2_edge[(j-1)*4 + 2],z2_edge[(j-1)*4 + 2]); 
//		d13 = distance(r1_edge[(j-1)*4 + 2],z1_edge[(j-1)*4 + 2],
//			       r3_edge[(j-1)*4 + 2],z3_edge[(j-1)*4 + 2]); 
//      		max1 = std::max(d12,d23);
//      		radius_edge[(j-1)*4 + 2] = std::max(max1,d13);
//		
//		d12 = distance(r1_edge[(j-1)*4 + 3],z1_edge[(j-1)*4 + 3],
//			       r2_edge[(j-1)*4 + 3],z2_edge[(j-1)*4 + 3]); 
//		d23 = distance(r3_edge[(j-1)*4 + 3],z3_edge[(j-1)*4 + 3],
//			       r2_edge[(j-1)*4 + 3],z2_edge[(j-1)*4 + 3]); 
//		d13 = distance(r1_edge[(j-1)*4 + 3],z1_edge[(j-1)*4 + 3],
//			       r3_edge[(j-1)*4 + 3],z3_edge[(j-1)*4 + 3]); 
//      		max1 = std::max(d12,d23);
//      		radius_edge[(j-1)*4 + 3] = std::max(max1,d13);
//      }
//
//    }
//  }
//
//  for(int ii = 0; ii< 4*ny;ii++)
//  {
//	  v1.push_back(v1_edge[ii]);
//	  v2.push_back(v2_edge[ii]);
//	  v3.push_back(v3_edge[ii]);
//  }
//    
//
//  return std::make_tuple(v1, v2, v3);
//}

int main()
{
    //cudaSetDevice(1);
    typedef std::chrono::high_resolution_clock app_time;
    auto app_start_time = app_time::now();
    
    std::cout << "Welcome to SOLPS processing for GITR!\n";
    
    std::vector<int> nxnyns = read_ifield("b2fstate","nx,ny,ns");
    
    // Global variables
    nx = nxnyns[0];
    ny = nxnyns[1];
    ns = nxnyns[2];
    
    int n8 = 8;
    int n_total = nx*ny*n8 + 4*ny;
   
    // SOLPS variables 
    std::vector<double> zamin = read_dfield("b2fstate","zamin");
    std::vector<double> zn = read_dfield("b2fstate","zn");
    std::vector<double> am = read_dfield("b2fstate","am");
    std::vector<double> na = read_dfield("b2fstate","na");
    std::vector<double> ne = read_dfield("b2fstate","ne");
    std::vector<double> fna = read_dfield("b2fstate","fna");
    std::vector<double> po = read_dfield("b2fstate", "po");
    std::vector<double> te = read_dfield("b2fstate", "te");
    std::vector<double> ti = read_dfield("b2fstate", "ti");
    std::vector<double> ua = read_dfield("b2fstate", "ua");
    std::vector<double> bb = read_dfield("b2fgmtry", "bb");
    std::vector<double> gs = read_dfield("b2fgmtry", "gs");
    for(int i=0; i<ti.size();i++)
    {
      ti[i] = ti[i]/1.602176565e-19;
      te[i] = te[i]/1.602176565e-19;
    }
    
    // SOLPS grid variables pulled out of the above variables
    std::vector<double> ion_density((nx+2)*(ny+2),0.0);
    std::vector<double> ion_charge((nx+2)*(ny+2),0.0);
    std::vector<double> ion_mass((nx+2)*(ny+2),0.0);
    std::vector<double> ion_flow((nx+2)*(ny+2),0.0);
    std::vector<double> ion_vr((nx+2)*(ny+2),0.0);
    std::vector<double> ion_vt((nx+2)*(ny+2),0.0);
    std::vector<double> ion_vz((nx+2)*(ny+2),0.0);
    std::vector<double> gradTer((nx+2)*(ny+2),0.0);
    std::vector<double> gradTet((nx+2)*(ny+2),0.0);
    std::vector<double> gradTez((nx+2)*(ny+2),0.0);
    std::vector<double> gradTir((nx+2)*(ny+2),0.0);
    std::vector<double> gradTit((nx+2)*(ny+2),0.0);
    std::vector<double> gradTiz((nx+2)*(ny+2),0.0);
    std::vector<std::vector<double>> flux(ns);
    std::vector<std::vector<double>> ni(ns);
    std::vector<std::vector<double>> flowv(ns);
    std::vector<std::vector<double>> bfield(4);
   
    for ( int i = 0 ; i < ns ; i++ )
    {
      flux[i].resize((nx+2)*(ny+2));
      ni[i].resize((nx+2)*(ny+2));
      flowv[i].resize((nx+2)*(ny+2));
    }
    
    for ( int i = 0 ; i < 4 ; i++ )
    {
      bfield[i].resize((nx+2)*(ny+2));
    }
    
    std::vector<double> Br, Bt, Bz, Bmag;
    std::tie(Br, Bt, Bz, Bmag) = get_Bfield(Br, Bt, Bz, Bmag);

    std::vector<double> gradTe, gradTi;
    std::tie(gradTe, gradTi) = get_gradT(gradTe,gradTi);


    for (int i=0;i<nx+2; i++)
    {
      for(int j=0; j<ny+2; j++)
      {
        for(int k=0; k<ns; k++)
        {
          flux[k][solps_2d_index(i,j)] = fna[solps_4d_index(i,j,0,k,2)];
          ni[k][solps_2d_index(i,j)] = na[solps_3d_index(i,j,k)];
          flowv[k][solps_2d_index(i,j)] = ua[solps_3d_index(i,j,k)];

          if(zamin[k] > 0.0)
          {	 
            ion_density[solps_2d_index(i,j)] = ion_density[solps_2d_index(i,j)] + na[solps_3d_index(i,j,k)];
            ion_charge[solps_2d_index(i,j)] = ion_charge[solps_2d_index(i,j)] + na[solps_3d_index(i,j,k)]*zamin[k];
            ion_mass[solps_2d_index(i,j)] = ion_mass[solps_2d_index(i,j)] + na[solps_3d_index(i,j,k)]*am[k];
            ion_flow[solps_2d_index(i,j)] = ion_flow[solps_2d_index(i,j)] + na[solps_3d_index(i,j,k)]*flowv[k][solps_2d_index(i,j)];
          }
        }
	if(ion_density[solps_2d_index(i,j)]>0.0)
	{
            ion_charge[solps_2d_index(i,j)] = ion_charge[solps_2d_index(i,j)]/ion_density[solps_2d_index(i,j)];
            ion_mass[solps_2d_index(i,j)] = ion_mass[solps_2d_index(i,j)]/ion_density[solps_2d_index(i,j)];
            
            ion_flow[solps_2d_index(i,j)] = ion_flow[solps_2d_index(i,j)]/ion_density[solps_2d_index(i,j)];
	}
	if(Bmag[solps_2d_index(i,j)] > 0.0)
	{
            ion_vr[solps_2d_index(i,j)] = ion_flow[solps_2d_index(i,j)]*Br[solps_2d_index(i,j)]/Bmag[solps_2d_index(i,j)];
            ion_vt[solps_2d_index(i,j)] = ion_flow[solps_2d_index(i,j)]*Bt[solps_2d_index(i,j)]/Bmag[solps_2d_index(i,j)];
            ion_vz[solps_2d_index(i,j)] = ion_flow[solps_2d_index(i,j)]*Bz[solps_2d_index(i,j)]/Bmag[solps_2d_index(i,j)];
            gradTer[solps_2d_index(i,j)] = gradTe[solps_2d_index(i,j)]*Br[solps_2d_index(i,j)]/Bmag[solps_2d_index(i,j)];
            gradTet[solps_2d_index(i,j)] = gradTe[solps_2d_index(i,j)]*Bt[solps_2d_index(i,j)]/Bmag[solps_2d_index(i,j)];
            gradTez[solps_2d_index(i,j)] = gradTe[solps_2d_index(i,j)]*Bz[solps_2d_index(i,j)]/Bmag[solps_2d_index(i,j)];
            gradTir[solps_2d_index(i,j)] = gradTi[solps_2d_index(i,j)]*Br[solps_2d_index(i,j)]/Bmag[solps_2d_index(i,j)];
            gradTit[solps_2d_index(i,j)] = gradTi[solps_2d_index(i,j)]*Bt[solps_2d_index(i,j)]/Bmag[solps_2d_index(i,j)];
            gradTiz[solps_2d_index(i,j)] = gradTi[solps_2d_index(i,j)]*Bz[solps_2d_index(i,j)]/Bmag[solps_2d_index(i,j)];
	}
      }	     
    } 
    
    // Target variables
    std::vector<double> r_inner_target(ny+1,0.0);
    std::vector<double> z_inner_target(ny+1,0.0);
    std::vector<double> area_inner_target(ny,0.0);
    std::vector<double> r_inner_target_midpoints(ny,0.0);
    std::vector<double> z_inner_target_midpoints(ny,0.0);
    std::vector<double> rmrs_inner_target(ny+1,0.0);
    std::vector<double> rmrs_inner_target_midpoints(ny,0.0);
    std::vector<double> Bmag_inner_target(ny,0.0);
    std::vector<double> Bangle_inner_target(ny,0.0);
    std::vector<double> te_inner_target(ny,0.0);
    std::vector<double> ti_inner_target(ny,0.0);
    std::vector<double> ne_inner_target(ny,0.0);
    std::vector<double> length_inner_target_segment(ny,0.0);
    
    std::vector<double> r_outer_target(ny+1,0.0);
    std::vector<double> z_outer_target(ny+1,0.0);
    std::vector<double> area_outer_target(ny,0.0);
    std::vector<double> r_outer_target_midpoints(ny,0.0);
    std::vector<double> z_outer_target_midpoints(ny,0.0);
    std::vector<double> rmrs_outer_target(ny+1,0.0);
    std::vector<double> rmrs_outer_target_midpoints(ny,0.0);
    std::vector<double> Bmag_outer_target(ny,0.0);
    std::vector<double> Bangle_outer_target(ny,0.0);
    std::vector<double> te_outer_target(ny,0.0);
    std::vector<double> ti_outer_target(ny,0.0);
    std::vector<double> ne_outer_target(ny,0.0);
    std::vector<double> length_outer_target_segment(ny,0.0);
    
    std::vector<double> flux_inner_target(ns*ny,0.0);
    std::vector<double> ni_inner_target(ns*ny,0.0);
    std::vector<double> flux_outer_target(ns*ny,0.0);
    std::vector<double> ni_outer_target(ns*ny,0.0);
    
    std::vector<double> crx = read_dfield("b2fgmtry", "crx");
    std::vector<double> cry = read_dfield("b2fgmtry", "cry");
    
    for (int j=0;j<ny+1; j++)
    {
      // Inner target	    
      int i = 0;	    
      double r_top_right = crx[solps_3d_index(i,j,3)];
      double z_top_right = cry[solps_3d_index(i,j,3)];
      
      r_inner_target[j] = r_top_right; 	    
      z_inner_target[j] = z_top_right; 	    
      
      if(j > 0)
      {
        double r_bottom_right = crx[solps_3d_index(i,j,1)];
        double z_bottom_right = cry[solps_3d_index(i,j,1)];
        r_inner_target_midpoints[j-1] = mean(r_top_right,r_bottom_right); 	    
        z_inner_target_midpoints[j-1] = mean(z_top_right,z_bottom_right); 	    
	length_inner_target_segment[j-1] = distance(r_top_right,z_top_right,r_bottom_right,z_bottom_right);
//	std::cout << " lits " << length_inner_target_segment[j-1] << std::endl;

        Bmag_inner_target[j-1] = Bmag[solps_2d_index(i+1,j)];

        // Bangle calculation
        double slope_dzdx = (z_top_right - z_bottom_right)/(r_top_right - r_bottom_right);
        double perpSlope = -std::copysign(1.0, slope_dzdx) / std::abs(slope_dzdx);
        double surface_normal[3] = {0.0,0.0,0.0};
        surface_normal[0] = 1.0 / std::sqrt(perpSlope * perpSlope + 1.0);
        surface_normal[2] = std::copysign(1.0,perpSlope) * std::sqrt(1.0 - surface_normal[0]*surface_normal[0]);
    
        double dot_product = surface_normal[0]*Br[solps_2d_index(i+1,j)] + surface_normal[2]*Bz[solps_2d_index(i+1,j)];
        double b_norm = Bmag_inner_target[j-1];
        Bangle_inner_target[j-1] = std::acos(dot_product/b_norm)*180.0/3.1415926535;
     
       	te_inner_target[j-1] = te[solps_2d_index(i+1,j)];
       	ti_inner_target[j-1] = ti[solps_2d_index(i+1,j)];
       	ne_inner_target[j-1] = ne[solps_2d_index(i+1,j)];

        area_inner_target[j-1] = gs[solps_3d_index(i+1,j,0)]; 	    
        for(int k=0; k<ns; k++)
        {
       	  flux_inner_target[k*ny + j-1] = flux[k][solps_2d_index(i+1,j)]/area_inner_target[j-1];
       	  ni_inner_target[k*ny + j-1] = na[solps_3d_index(i+1,j,k)];
	}
      }
      
      // Outer target
      i = nx;
      r_top_right = crx[solps_3d_index(i,j,3)];
      z_top_right = cry[solps_3d_index(i,j,3)];
      
      r_outer_target[j] = r_top_right; 	    
      z_outer_target[j] = z_top_right; 	    
      
      if(j > 0)
      {
        double r_bottom_right = crx[solps_3d_index(i,j,1)];
        double z_bottom_right = cry[solps_3d_index(i,j,1)];
        r_outer_target_midpoints[j-1] = mean(r_top_right,r_bottom_right); 	    
        z_outer_target_midpoints[j-1] = mean(z_top_right,z_bottom_right); 	    
	length_outer_target_segment[j-1] = distance(r_top_right,z_top_right,r_bottom_right,z_bottom_right);
        
	Bmag_outer_target[j-1] = Bmag[solps_2d_index(i,j)];

        // Bangle calculation
        double slope_dzdx = (z_top_right - z_bottom_right)/(r_top_right - r_bottom_right);
        double perpSlope = -std::copysign(1.0, slope_dzdx) / std::abs(slope_dzdx);
        double surface_normal[3] = {0.0,0.0,0.0};
        surface_normal[0] = 1.0 / std::sqrt(perpSlope * perpSlope + 1.0);
        surface_normal[2] = std::copysign(1.0,perpSlope) * std::sqrt(1.0 - surface_normal[0]*surface_normal[0]);
    
        double dot_product = surface_normal[0]*Br[solps_2d_index(i,j)] + surface_normal[2]*Bz[solps_2d_index(i,j)];
        double b_norm = Bmag_outer_target[j-1];
        Bangle_outer_target[j-1] = std::acos(dot_product/b_norm)*180.0/3.1415926535;
     
       	te_outer_target[j-1] = te[solps_2d_index(i,j)];
       	ti_outer_target[j-1] = ti[solps_2d_index(i,j)];
       	ne_outer_target[j-1] = ne[solps_2d_index(i,j)];

        area_outer_target[j-1] = gs[solps_3d_index(i+1,j,0)]; 	    
        for(int k=0; k<ns; k++)
        {
       	  flux_outer_target[k*ny + j-1] = flux[k][solps_2d_index(i+1,j)]/area_outer_target[j-1];
       	  ni_outer_target[k*ny + j-1] = na[solps_3d_index(i,j,k)];
	}
      }
    }

    std::vector<int> topcut_vec = read_ifield("b2fgmtry","topcut");
    int topcut = topcut_vec[0];//Doesn't change from SOLPS index value because of removal of guard cell

    for(int i=topcut; i<ny+1; i++)
    {
      if(i==topcut)
      {
        rmrs_inner_target[i] = 0.0;
        rmrs_outer_target[i] = 0.0;
        rmrs_inner_target_midpoints[i] = 0.5*rmrs_inner_target[i];
        rmrs_outer_target_midpoints[i] = 0.5*rmrs_outer_target[i];
      }
      else
      {
        rmrs_inner_target[i] = rmrs_inner_target[i-1] + length_inner_target_segment[i-1];
        rmrs_outer_target[i] = rmrs_outer_target[i-1] + length_outer_target_segment[i-1];
	if(i<ny)
	{
          rmrs_inner_target_midpoints[i] = rmrs_inner_target[i] + 0.5*length_inner_target_segment[i];
          rmrs_outer_target_midpoints[i] = rmrs_outer_target[i] + 0.5*length_outer_target_segment[i];
	}
      }
    }
    
    for(int i=topcut-1; i>-1; i--)
    {
        rmrs_inner_target[i] = rmrs_inner_target[i+1] - length_inner_target_segment[i];
        rmrs_outer_target[i] = rmrs_outer_target[i+1] - length_outer_target_segment[i];
        rmrs_inner_target_midpoints[i] = rmrs_inner_target[i] - 0.5*length_inner_target_segment[i];
        rmrs_outer_target_midpoints[i] = rmrs_outer_target[i] - 0.5*length_outer_target_segment[i];
    }

    auto solps_fields = new Fields();

    thrust::host_vector<double> r1_h(n_total), r2_h(n_total), r3_h(n_total),
                        z1_h(n_total), z2_h(n_total), z3_h(n_total),
                        v1_h(n_total), v2_h(n_total), v3_h(n_total),
                        radius_h(n_total);

    // Calculate Efield on SOLPS grid
    std::vector<double> Er, Ez;
    std::vector<double> r1t, r2t, r3t, z1t, z2t, z3t, po1t, po2t, po3t, radiust;
    std::tie(po1t, po2t, po3t) = get_scalar_field_tris(po,po1t, po2t, po3t);
    std::tie(r1t, r2t, r3t, z1t, z2t, z3t,radiust) = get_triangles(r1t, r2t, r3t, z1t, z2t, z3t, radiust);

    std::tie(Er, Ez) = get_Efield(Er, Ez);
    
    std::vector<double> mass1t, mass2t, mass3t;
    std::tie(mass1t, mass2t, mass3t) = get_scalar_field_tris(ion_mass,mass1t, mass2t, mass3t);
    solps_fields->mass1t.resize(mass1t.size());
    solps_fields->mass1t = mass1t;
    solps_fields->mass2t.resize(mass2t.size());
    solps_fields->mass2t = mass2t;
    solps_fields->mass3t.resize(mass3t.size());
    solps_fields->mass3t = mass3t;
    
    std::vector<double> charge1t, charge2t, charge3t;
    std::tie(charge1t, charge2t, charge3t) = get_scalar_field_tris(ion_charge,charge1t, charge2t, charge3t);
    solps_fields->charge1t.resize(charge1t.size());
    solps_fields->charge1t = charge1t;
    solps_fields->charge2t.resize(charge2t.size());
    solps_fields->charge2t = charge2t;
    solps_fields->charge3t.resize(charge3t.size());
    solps_fields->charge3t = charge3t;
    
    std::vector<double> Br1t, Br2t, Br3t;
    std::tie(Br1t, Br2t, Br3t) = get_scalar_field_tris(Br,Br1t, Br2t, Br3t);
    solps_fields->Br1t.resize(Br1t.size());
    solps_fields->Br1t = Br1t;
    solps_fields->Br2t.resize(Br2t.size());
    solps_fields->Br2t = Br2t;
    solps_fields->Br3t.resize(Br3t.size());
    solps_fields->Br3t = Br3t;
    
    std::vector<double> Bt1t, Bt2t, Bt3t;
    std::tie(Bt1t, Bt2t, Bt3t) = get_scalar_field_tris(Bt,Bt1t, Bt2t, Bt3t);
    solps_fields->Bt1t.resize(Bt1t.size());
    solps_fields->Bt1t = Bt1t;
    solps_fields->Bt2t.resize(Bt2t.size());
    solps_fields->Bt2t = Bt2t;
    solps_fields->Bt3t.resize(Bt3t.size());
    solps_fields->Bt3t = Bt3t;
    
    std::vector<double> Bz1t, Bz2t, Bz3t;
    std::tie(Bz1t, Bz2t, Bz3t) = get_scalar_field_tris(Bz,Bz1t, Bz2t, Bz3t);
    solps_fields->Bz1t.resize(Bz1t.size());
    solps_fields->Bz1t = Bz1t;
    solps_fields->Bz2t.resize(Bz2t.size());
    solps_fields->Bz2t = Bz2t;
    solps_fields->Bz3t.resize(Bz3t.size());
    solps_fields->Bz3t = Bz3t;
    
    std::vector<double> Bmag1t, Bmag2t, Bmag3t;
    std::tie(Bmag1t, Bmag2t, Bmag3t) = get_scalar_field_tris(Bmag,Bmag1t, Bmag2t, Bmag3t);
    solps_fields->Bmag1t.resize(Bmag1t.size());
    solps_fields->Bmag1t = Bmag1t;
    solps_fields->Bmag2t.resize(Bmag2t.size());
    solps_fields->Bmag2t = Bmag2t;
    solps_fields->Bmag3t.resize(Bmag3t.size());
    solps_fields->Bmag3t = Bmag3t;
    
    std::vector<double> te1t, te2t, te3t;
    std::tie(te1t, te2t, te3t) = get_scalar_field_tris(te,te1t, te2t, te3t);
    solps_fields->te1t.resize(te1t.size());
    solps_fields->te1t = te1t;
    solps_fields->te2t.resize(te2t.size());
    solps_fields->te2t = te2t;
    solps_fields->te3t.resize(te3t.size());
    solps_fields->te3t = te3t;
    
    std::vector<double> ti1t, ti2t, ti3t;
    std::tie(ti1t, ti2t, ti3t) = get_scalar_field_tris(ti,ti1t, ti2t, ti3t);
    solps_fields->ti1t.resize(ti1t.size());
    solps_fields->ti1t = ti1t;
    solps_fields->ti2t.resize(ti2t.size());
    solps_fields->ti2t = ti2t;
    solps_fields->ti3t.resize(ti3t.size());
    solps_fields->ti3t = ti3t;
    //std::vector<double> flux_last1t, flux_last2t, flux_last3t;
    //std::tie(flux_last1t, flux_last2t, flux_last3t) = get_scalar_field_tris_left(flux_last,flux_last1t, flux_last2t, flux_last3t);
    //solps_fields->flux_last1t.resize(flux_last1t.size());
    //solps_fields->flux_last1t = flux_last1t;
    //solps_fields->flux_last2t.resize(flux_last2t.size());
    //solps_fields->flux_last2t = flux_last2t;
    //solps_fields->flux_last3t.resize(flux_last3t.size());
    //solps_fields->flux_last3t = flux_last3t;
    std::cout << " size of po1t " << po1t.size() << std::endl;
    std::cout << " size of r1t " << r1t.size() << std::endl;
    std::cout << " size of te1t " << te1t.size() << std::endl;
    
    std::vector<double> ni1t, ni2t, ni3t;
    std::tie(ni1t, ni2t, ni3t) = get_scalar_field_tris(ion_density, ni1t, ni2t, ni3t);
    solps_fields->ni1t.resize(ni1t.size());
    solps_fields->ni1t = ni1t;
    solps_fields->ni2t.resize(ni2t.size());
    solps_fields->ni2t = ni2t;
    solps_fields->ni3t.resize(ni3t.size());
    solps_fields->ni3t = ni3t;
    
    std::vector<double> ne1t, ne2t, ne3t;
    std::tie(ne1t, ne2t, ne3t) = get_scalar_field_tris(ne, ne1t, ne2t, ne3t);
    solps_fields->ne1t.resize(ne1t.size());
    solps_fields->ne1t = ne1t;
    solps_fields->ne2t.resize(ne2t.size());
    solps_fields->ne2t = ne2t;
    solps_fields->ne3t.resize(ne3t.size());
    solps_fields->ne3t = ne3t;
    
    std::vector<double> vr1t, vr2t, vr3t;
    std::tie(vr1t, vr2t, vr3t) = get_scalar_field_tris(ion_vr,vr1t, vr2t, vr3t);
    solps_fields->vr1t.resize(vr1t.size());
    solps_fields->vr1t = vr1t;
    solps_fields->vr2t.resize(vr2t.size());
    solps_fields->vr2t = vr2t;
    solps_fields->vr3t.resize(vr3t.size());
    solps_fields->vr3t = vr3t;

    std::vector<double> vt1t, vt2t, vt3t;
    std::tie(vt1t, vt2t, vt3t) = get_scalar_field_tris(ion_vt,vt1t, vt2t, vt3t);
    solps_fields->vt1t.resize(vt1t.size());
    solps_fields->vt1t = vt1t;
    solps_fields->vt2t.resize(vt2t.size());
    solps_fields->vt2t = vt2t;
    solps_fields->vt3t.resize(vt3t.size());
    solps_fields->vt3t = vt3t;
    
    std::vector<double> vz1t, vz2t, vz3t;
    std::tie(vz1t, vz2t, vz3t) = get_scalar_field_tris(ion_vz,vz1t, vz2t, vz3t);
    solps_fields->vz1t.resize(vz1t.size());
    solps_fields->vz1t = vz1t;
    solps_fields->vz2t.resize(vz2t.size());
    solps_fields->vz2t = vz2t;
    solps_fields->vz3t.resize(vz3t.size());
    solps_fields->vz3t = vz3t;
    
    std::vector<double> Er1t, Er2t, Er3t;
    std::tie(Er1t, Er2t, Er3t) = get_scalar_field_tris(Er,Er1t, Er2t, Er3t);
    solps_fields->Er1t.resize(Er1t.size());
    solps_fields->Er1t = Er1t;
    solps_fields->Er2t.resize(Er2t.size());
    solps_fields->Er2t = Er2t;
    solps_fields->Er3t.resize(Er3t.size());
    solps_fields->Er3t = Er3t;

    std::vector<double> Ez1t, Ez2t, Ez3t;
    std::tie(Ez1t, Ez2t, Ez3t) = get_scalar_field_tris(Ez,Ez1t, Ez2t, Ez3t);
    solps_fields->Ez1t.resize(Ez1t.size());
    solps_fields->Ez1t = Ez1t;
    solps_fields->Ez2t.resize(Ez2t.size());
    solps_fields->Ez2t = Ez2t;
    solps_fields->Ez3t.resize(Ez3t.size());
    solps_fields->Ez3t = Ez3t;
    
    
    std::vector<double> gradTe1t, gradTe2t, gradTe3t;
    std::tie(gradTe1t, gradTe2t, gradTe3t) = get_scalar_field_tris(gradTe,gradTe1t, gradTe2t, gradTe3t);
    solps_fields->gradTe1t.resize(gradTe1t.size());
    solps_fields->gradTe1t = gradTe1t;
    solps_fields->gradTe2t.resize(gradTe2t.size());
    solps_fields->gradTe2t = gradTe2t;
    solps_fields->gradTe3t.resize(gradTe3t.size());
    solps_fields->gradTe3t = gradTe3t;
    
    std::vector<double> gradTer1t, gradTer2t, gradTer3t;
    std::tie(gradTer1t, gradTer2t, gradTer3t) = get_scalar_field_tris(gradTer,gradTer1t, gradTer2t, gradTer3t);
    solps_fields->gradTer1t.resize(gradTer1t.size());
    solps_fields->gradTer1t = gradTer1t;
    solps_fields->gradTer2t.resize(gradTer2t.size());
    solps_fields->gradTer2t = gradTer2t;
    solps_fields->gradTer3t.resize(gradTer3t.size());
    solps_fields->gradTer3t = gradTer3t;

    std::vector<double> gradTet1t, gradTet2t, gradTet3t;
    std::tie(gradTet1t, gradTet2t, gradTet3t) = get_scalar_field_tris(gradTet,gradTet1t, gradTet2t, gradTet3t);
    solps_fields->gradTet1t.resize(gradTet1t.size());
    solps_fields->gradTet1t = gradTet1t;
    solps_fields->gradTet2t.resize(gradTet2t.size());
    solps_fields->gradTet2t = gradTet2t;
    solps_fields->gradTet3t.resize(gradTet3t.size());
    solps_fields->gradTet3t = gradTet3t;
    
    std::vector<double> gradTez1t, gradTez2t, gradTez3t;
    std::tie(gradTez1t, gradTez2t, gradTez3t) = get_scalar_field_tris(gradTez,gradTez1t, gradTez2t, gradTez3t);
    solps_fields->gradTez1t.resize(gradTez1t.size());
    solps_fields->gradTez1t = gradTez1t;
    solps_fields->gradTez2t.resize(gradTez2t.size());
    solps_fields->gradTez2t = gradTez2t;
    solps_fields->gradTez3t.resize(gradTez3t.size());
    solps_fields->gradTez3t = gradTez3t;
    
    std::vector<double> gradTi1t, gradTi2t, gradTi3t;
    std::tie(gradTi1t, gradTi2t, gradTi3t) = get_scalar_field_tris(gradTi,gradTi1t, gradTi2t, gradTi3t);
    solps_fields->gradTi1t.resize(gradTi1t.size());
    solps_fields->gradTi1t = gradTi1t;
    solps_fields->gradTi2t.resize(gradTi2t.size());
    solps_fields->gradTi2t = gradTi2t;
    solps_fields->gradTi3t.resize(gradTi3t.size());
    solps_fields->gradTi3t = gradTi3t;
    
    std::vector<double> gradTir1t, gradTir2t, gradTir3t;
    std::tie(gradTir1t, gradTir2t, gradTir3t) = get_scalar_field_tris(gradTir,gradTir1t, gradTir2t, gradTir3t);
    solps_fields->gradTir1t.resize(gradTir1t.size());
    solps_fields->gradTir1t = gradTir1t;
    solps_fields->gradTir2t.resize(gradTir2t.size());
    solps_fields->gradTir2t = gradTir2t;
    solps_fields->gradTir3t.resize(gradTir3t.size());
    solps_fields->gradTir3t = gradTir3t;

    std::vector<double> gradTit1t, gradTit2t, gradTit3t;
    std::tie(gradTit1t, gradTit2t, gradTit3t) = get_scalar_field_tris(gradTit,gradTit1t, gradTit2t, gradTit3t);
    solps_fields->gradTit1t.resize(gradTit1t.size());
    solps_fields->gradTit1t = gradTit1t;
    solps_fields->gradTit2t.resize(gradTit2t.size());
    solps_fields->gradTit2t = gradTit2t;
    solps_fields->gradTit3t.resize(gradTit3t.size());
    solps_fields->gradTit3t = gradTit3t;
    
    std::vector<double> gradTiz1t, gradTiz2t, gradTiz3t;
    std::tie(gradTiz1t, gradTiz2t, gradTiz3t) = get_scalar_field_tris(gradTiz,gradTiz1t, gradTiz2t, gradTiz3t);
    solps_fields->gradTiz1t.resize(gradTiz1t.size());
    solps_fields->gradTiz1t = gradTiz1t;
    solps_fields->gradTiz2t.resize(gradTiz2t.size());
    solps_fields->gradTiz2t = gradTiz2t;
    solps_fields->gradTiz3t.resize(gradTiz3t.size());
    solps_fields->gradTiz3t = gradTiz3t;
    netCDF::NcFile ncFile_tri("solps_triangles2.nc",
                         netCDF::NcFile::replace);
      netCDF::NcDim _ntri = ncFile_tri.addDim("ntri", nx*ny*8 + ny*4);
      //netCDF::NcDim _nyy = ncFile_tri.addDim("ny", ny);
      //netCDF::NcDim _n8 = ncFile_tri.addDim("n8", n8);
      //netCDF::NcDim _ntot = ncFile_tri.addDim("ntot", n_total);
      //std::vector<netCDF::NcDim> outdimt;
      //outdimt.push_back(_n8);
      //outdimt.push_back(_nyy);
      //outdimt.push_back(_nxx);

      netCDF::NcVar _r1 = ncFile_tri.addVar("r1", netCDF::ncDouble, _ntri);
      netCDF::NcVar _r2 = ncFile_tri.addVar("r2", netCDF::ncDouble, _ntri);
      netCDF::NcVar _r3 = ncFile_tri.addVar("r3", netCDF::ncDouble, _ntri);
      netCDF::NcVar _z1 = ncFile_tri.addVar("z1", netCDF::ncDouble, _ntri);
      netCDF::NcVar _z2 = ncFile_tri.addVar("z2", netCDF::ncDouble, _ntri);
      netCDF::NcVar _z3 = ncFile_tri.addVar("z3", netCDF::ncDouble, _ntri);
      netCDF::NcVar _v1 = ncFile_tri.addVar("v1", netCDF::ncDouble, _ntri);
      netCDF::NcVar _v2 = ncFile_tri.addVar("v2", netCDF::ncDouble, _ntri);
      netCDF::NcVar _v3 = ncFile_tri.addVar("v3", netCDF::ncDouble, _ntri);
      netCDF::NcVar _ni1 = ncFile_tri.addVar("ni1", netCDF::ncDouble, _ntri);
      netCDF::NcVar _ni2 = ncFile_tri.addVar("ni2", netCDF::ncDouble, _ntri);
      netCDF::NcVar _ni3 = ncFile_tri.addVar("ni3", netCDF::ncDouble, _ntri);
      netCDF::NcVar _bt1 = ncFile_tri.addVar("bt1", netCDF::ncDouble, _ntri);
      netCDF::NcVar _bt2 = ncFile_tri.addVar("bt2", netCDF::ncDouble, _ntri);
      netCDF::NcVar _bt3 = ncFile_tri.addVar("bt3", netCDF::ncDouble, _ntri);
      netCDF::NcVar _radius = ncFile_tri.addVar("radius", netCDF::ncDouble, _ntri);
      _r1.putVar(&r1t[0]);
      _r2.putVar(&r2t[0]);
      _r3.putVar(&r3t[0]);

      _z1.putVar(&z1t[0]);
      _z2.putVar(&z2t[0]);
      _z3.putVar(&z3t[0]);
      _v1.putVar(&po1t[0]);
      _v2.putVar(&po2t[0]);
      _v3.putVar(&po3t[0]);
      _ni1.putVar(solps_fields->ni1t.data());
      _ni2.putVar(solps_fields->ni2t.data());
      _ni3.putVar(solps_fields->ni3t.data());
      _bt1.putVar(solps_fields->Bt1t.data());
      _bt2.putVar(solps_fields->Bt2t.data());
      _bt3.putVar(solps_fields->Bt3t.data());
      _radius.putVar(&radiust[0]);
      ncFile_tri.close();
    
    thrust::device_vector<double> r1(r1t);
    thrust::device_vector<double> r2(r2t);
    thrust::device_vector<double> r3(r3t);
    thrust::device_vector<double> z1(z1t);
    thrust::device_vector<double> z2(z2t);
    thrust::device_vector<double> z3(z3t);
    thrust::device_vector<double> v1(po1t);
    thrust::device_vector<double> v2(po2t);
    thrust::device_vector<double> v3(po3t);
    thrust::device_vector<double> radius(radiust);
 
    double* r1_pointer = thrust::raw_pointer_cast(&r1[0]);
    double* z1_pointer = thrust::raw_pointer_cast(&z1[0]);
    double* v1_pointer = thrust::raw_pointer_cast(&v1[0]);
    double* r2_pointer = thrust::raw_pointer_cast(&r2[0]);
    double* z2_pointer = thrust::raw_pointer_cast(&z2[0]);
    double* v2_pointer = thrust::raw_pointer_cast(&v2[0]);
    double* r3_pointer = thrust::raw_pointer_cast(&r3[0]);
    double* z3_pointer = thrust::raw_pointer_cast(&z3[0]);
    double* v3_pointer = thrust::raw_pointer_cast(&v3[0]);
    double* radius_pointer = thrust::raw_pointer_cast(&radius[0]);
    
    int nr = 4400;
    int nz = 9300;

    solps_fields->te.resize(nr*nz);
    solps_fields->te = 0.0;
    solps_fields->ti.resize(nr*nz);
    solps_fields->ti = 0.0;
    solps_fields->ni.resize(nr*nz);
    solps_fields->ni = 0.0;
    solps_fields->ne.resize(nr*nz);
    solps_fields->ne = 0.0;
    //solps_fields->flux_last.resize(nr*nz);
    //solps_fields->flux_last = 0.0;
    solps_fields->mass.resize(nr*nz);
    solps_fields->mass = 0.0;
    solps_fields->charge.resize(nr*nz);
    solps_fields->charge = 0.0;
    solps_fields->Br.resize(nr*nz);
    solps_fields->Br = 0.0;
    solps_fields->Bt.resize(nr*nz);
    solps_fields->Bt = 0.0;
    solps_fields->Bz.resize(nr*nz);
    solps_fields->Bz = 0.0;
    solps_fields->Bmag.resize(nr*nz);
    solps_fields->Bmag = 0.0;
    solps_fields->vr.resize(nr*nz);
    solps_fields->vr = 0.0;
    solps_fields->vt.resize(nr*nz);
    solps_fields->vt = 0.0;
    solps_fields->vz.resize(nr*nz);
    solps_fields->vz = 0.0;
    solps_fields->Er.resize(nr*nz);
    solps_fields->Er = 0.0;
    solps_fields->Ez.resize(nr*nz);
    solps_fields->Ez = 0.0;
    std::vector<double> Et(nr*nz,0.0);
    solps_fields->gradTe.resize(nr*nz);
    solps_fields->gradTe = 0.0;
    solps_fields->gradTer.resize(nr*nz);
    solps_fields->gradTer = 0.0;
    solps_fields->gradTet.resize(nr*nz);
    solps_fields->gradTet = 0.0;
    solps_fields->gradTez.resize(nr*nz);
    solps_fields->gradTez = 0.0;
    solps_fields->gradTi.resize(nr*nz);
    solps_fields->gradTi = 0.0;
    solps_fields->gradTir.resize(nr*nz);
    solps_fields->gradTir = 0.0;
    solps_fields->gradTit.resize(nr*nz);
    solps_fields->gradTit = 0.0;
    solps_fields->gradTiz.resize(nr*nz);
    solps_fields->gradTiz = 0.0;

    thrust::counting_iterator<std::size_t> point_first(0);
    thrust::counting_iterator<std::size_t> point_last(nr*nz);
    double r_start = 4.0; //4.5018; //4.0;
    double r_end = 8.5; //4.5018; //8.4;
    double z_start = -4.6; //-3.131; //-4.6;
    double z_end = 4.7; //-3.131; //4.7;

    double dr = (r_end - r_start)/(nr - 1);
    double dz = (z_end - z_start)/(nz - 1);

    if(nr == 1)
    {
	    dr = 1;
    }
    if(nz == 1)
    {
	    dz = 1;
    }
    thrust::host_vector<double> r_h(nr), z_h(nz), val_h(nr*nz);

    for(int i=0; i< nr; i++)
    {
      r_h[i] = r_start + i*dr;
    }
    
    for(int j=0; j < nz; j++)
    {
      z_h[j] = z_start + j*dz;
    }
    
    thrust::device_vector<double> r = r_h;
    thrust::device_vector<double> z = z_h; 
    thrust::device_vector<double> val(nr*nz,0.0);
    thrust::device_vector<bool> found(nr*nz,true);
    double* r_pointer = thrust::raw_pointer_cast(&r[0]);
    double* z_pointer = thrust::raw_pointer_cast(&z[0]);
    double* val_pointer = thrust::raw_pointer_cast(&val[0]);
    bool* found_pointer = thrust::raw_pointer_cast(&found[0]);
    
    //int nr = 1;
    //int nz = 1;
    //std::vector<double> r(1,6.0);
    //std::vector<double> z(1,0.0);
    //std::vector<double> val(1,0.0);
    thrust::device_vector<int> index_triangle(1,0);
    int* triangle_pointer = thrust::raw_pointer_cast(&index_triangle[0]);
    saxpy_functor spf(triangle_pointer,nr,nz,r_pointer,z_pointer,
                      r1_pointer,z1_pointer,v1_pointer,
                      r2_pointer,z2_pointer,v2_pointer,
                      r3_pointer,z3_pointer,v3_pointer,
                      radius_pointer,val_pointer,found_pointer, solps_fields);
    
    auto setup_time_clock = app_time::now();
    typedef std::chrono::duration<float> fsec;
    fsec setup_time = setup_time_clock - app_start_time;
    printf("Time taken for setup is %6.3f (secs) \n", setup_time.count());
//#pragma omp parallel for collapse(3)
  for(int i=0; i< n_total; i++)
  {
    //for(int jj=0; jj < nz; jj++)
    //{
    //  for(int ii=0; ii< nr; ii++)
    //  {
          index_triangle[0] = i;
          thrust::for_each(thrust::device,point_first,point_last,spf);
              //double dist = (r[ii] - r1[i])*(r[ii] - r1[i]) + (z[jj] - z1[i])*(z[jj] - z1[i]);
              //if(dist < radius[i]*radius[i])
              //{
              //  bool in = in_triangle(r1[i],z1[i],r2[i],z2[i],r3[i],z3[i],r[ii],z[jj]);

              //  if(in)
              //  {
              //    double a_point = interpolate_value(r1[i],z1[i],v1[i],
              //                                     r2[i],z2[i],v2[i],
              //                                     r3[i],z3[i],v3[i],
              //                                     r[ii],z[jj]);
              //    val[jj*nr + ii] = a_point;
              //  }
              //}
    //      }
    //    }
      
    }
    auto run_time_clock = app_time::now();
    //typedef std::chrono::duration<float> fsec;
    fsec run_time = run_time_clock - setup_time_clock;
    printf("Time taken for main loop is %6.3f (secs) \n", run_time.count());

    val_h = val;

      netCDF::NcFile ncFile_out("interpolated_values.nc",
                         netCDF::NcFile::replace);
      netCDF::NcDim _nr = ncFile_out.addDim("nr", nr);
      netCDF::NcDim _nz = ncFile_out.addDim("nz", nz);
      netCDF::NcDim _ny = ncFile_out.addDim("ny", ny);
      netCDF::NcDim _ns = ncFile_out.addDim("ns", ns);
      netCDF::NcDim _ny1 = ncFile_out.addDim("ny+1", ny+1);
      std::vector<netCDF::NcDim> outdim;
      outdim.push_back(_nz);
      outdim.push_back(_nr);
      
      std::vector<netCDF::NcDim> surfdim;
      surfdim.push_back(_ns);
      surfdim.push_back(_ny);


      netCDF::NcVar _z_spec = ncFile_out.addVar("atomic_number", netCDF::ncDouble, _ns);
      netCDF::NcVar _q_spec = ncFile_out.addVar("charge_number", netCDF::ncDouble, _ns);
      netCDF::NcVar _a_spec = ncFile_out.addVar("mass_number", netCDF::ncDouble, _ns);
      
      netCDF::NcVar _r_inner_target = ncFile_out.addVar("r_inner_target", netCDF::ncDouble, _ny1);
      netCDF::NcVar _z_inner_target = ncFile_out.addVar("z_inner_target", netCDF::ncDouble, _ny1);
      netCDF::NcVar _r_inner_target_midpoints = ncFile_out.addVar("r_inner_target_midpoints", netCDF::ncDouble, _ny);
      netCDF::NcVar _z_inner_target_midpoints = ncFile_out.addVar("z_inner_target_midpoints", netCDF::ncDouble, _ny);
      netCDF::NcVar _rmrs_inner_target = ncFile_out.addVar("rmrs_inner_target", netCDF::ncDouble, _ny1);
      netCDF::NcVar _rmrs_inner_target_midpoints = ncFile_out.addVar("rmrs_inner_target_midpoints", netCDF::ncDouble, _ny);
      netCDF::NcVar _Bmag_inner_target = ncFile_out.addVar("Bmag_inner_target", netCDF::ncDouble, _ny);
      netCDF::NcVar _Bangle_inner_target = ncFile_out.addVar("Bangle_inner_target", netCDF::ncDouble, _ny);
      netCDF::NcVar _te_inner_target = ncFile_out.addVar("te_inner_target", netCDF::ncDouble, _ny);
      netCDF::NcVar _ti_inner_target = ncFile_out.addVar("ti_inner_target", netCDF::ncDouble, _ny);
      netCDF::NcVar _ne_inner_target = ncFile_out.addVar("ne_inner_target", netCDF::ncDouble, _ny);
      netCDF::NcVar _flux_inner_target = ncFile_out.addVar("flux_inner_target", netCDF::ncDouble, surfdim);
      netCDF::NcVar _ni_inner_target = ncFile_out.addVar("ni_inner_target", netCDF::ncDouble, surfdim);

      netCDF::NcVar _r_outer_target = ncFile_out.addVar("r_outer_target", netCDF::ncDouble, _ny1);
      netCDF::NcVar _z_outer_target = ncFile_out.addVar("z_outer_target", netCDF::ncDouble, _ny1);
      netCDF::NcVar _r_outer_target_midpoints = ncFile_out.addVar("r_outer_target_midpoints", netCDF::ncDouble, _ny);
      netCDF::NcVar _z_outer_target_midpoints = ncFile_out.addVar("z_outer_target_midpoints", netCDF::ncDouble, _ny);
      netCDF::NcVar _rmrs_outer_target = ncFile_out.addVar("rmrs_outer_target", netCDF::ncDouble, _ny1);
      netCDF::NcVar _rmrs_outer_target_midpoints = ncFile_out.addVar("rmrs_outer_target_midpoints", netCDF::ncDouble, _ny);
      netCDF::NcVar _Bmag_outer_target = ncFile_out.addVar("Bmag_outer_target", netCDF::ncDouble, _ny);
      netCDF::NcVar _Bangle_outer_target = ncFile_out.addVar("Bangle_outer_target", netCDF::ncDouble, _ny);
      netCDF::NcVar _te_outer_target = ncFile_out.addVar("te_outer_target", netCDF::ncDouble, _ny);
      netCDF::NcVar _ti_outer_target = ncFile_out.addVar("ti_outer_target", netCDF::ncDouble, _ny);
      netCDF::NcVar _ne_outer_target = ncFile_out.addVar("ne_outer_target", netCDF::ncDouble, _ny);
      netCDF::NcVar _flux_outer_target = ncFile_out.addVar("flux_outer_target", netCDF::ncDouble, surfdim);
      netCDF::NcVar _ni_outer_target = ncFile_out.addVar("ni_outer_target", netCDF::ncDouble, surfdim);

      netCDF::NcVar _gridr = ncFile_out.addVar("gridr", netCDF::ncDouble, _nr);
      netCDF::NcVar _gridz = ncFile_out.addVar("gridz", netCDF::ncDouble, _nz);
      netCDF::NcVar _vals = ncFile_out.addVar("values", netCDF::ncDouble, outdim);
      netCDF::NcVar _te = ncFile_out.addVar("te", netCDF::ncDouble, outdim);
      netCDF::NcVar _ti = ncFile_out.addVar("ti", netCDF::ncDouble, outdim);
      netCDF::NcVar _ne = ncFile_out.addVar("ne", netCDF::ncDouble, outdim);
      netCDF::NcVar _ni = ncFile_out.addVar("ni", netCDF::ncDouble, outdim);
      netCDF::NcVar _flux = ncFile_out.addVar("flux", netCDF::ncDouble, outdim);
      netCDF::NcVar _mass = ncFile_out.addVar("mass", netCDF::ncDouble, outdim);
      netCDF::NcVar _charge = ncFile_out.addVar("charge", netCDF::ncDouble, outdim);
      netCDF::NcVar _Br = ncFile_out.addVar("Br", netCDF::ncDouble, outdim);
      netCDF::NcVar _Bt = ncFile_out.addVar("Bt", netCDF::ncDouble, outdim);
      netCDF::NcVar _Bz = ncFile_out.addVar("Bz", netCDF::ncDouble, outdim);
      netCDF::NcVar _Bmag = ncFile_out.addVar("Bmag", netCDF::ncDouble, outdim);
      netCDF::NcVar _vr = ncFile_out.addVar("vr", netCDF::ncDouble, outdim);
      netCDF::NcVar _vt = ncFile_out.addVar("vt", netCDF::ncDouble, outdim);
      netCDF::NcVar _vz = ncFile_out.addVar("vz", netCDF::ncDouble, outdim);
      netCDF::NcVar _Er = ncFile_out.addVar("Er", netCDF::ncDouble, outdim);
      netCDF::NcVar _Ez = ncFile_out.addVar("Ez", netCDF::ncDouble, outdim);
      netCDF::NcVar _Et = ncFile_out.addVar("Et", netCDF::ncDouble, outdim);
      netCDF::NcVar _gradTe = ncFile_out.addVar("gradTe", netCDF::ncDouble, outdim);
      netCDF::NcVar _gradTer = ncFile_out.addVar("gradTer", netCDF::ncDouble, outdim);
      netCDF::NcVar _gradTet = ncFile_out.addVar("gradTet", netCDF::ncDouble, outdim);
      netCDF::NcVar _gradTez = ncFile_out.addVar("gradTez", netCDF::ncDouble, outdim);
      netCDF::NcVar _gradTi = ncFile_out.addVar("gradTi", netCDF::ncDouble, outdim);
      netCDF::NcVar _gradTir = ncFile_out.addVar("gradTir", netCDF::ncDouble, outdim);
      netCDF::NcVar _gradTit = ncFile_out.addVar("gradTit", netCDF::ncDouble, outdim);
      netCDF::NcVar _gradTiz = ncFile_out.addVar("gradTiz", netCDF::ncDouble, outdim);

      _z_spec.putVar(&zn[0]);
      _q_spec.putVar(&zamin[0]);
      _a_spec.putVar(&am[0]);

      _r_inner_target.putVar(&r_inner_target[0]);
      _z_inner_target.putVar(&z_inner_target[0]);
      _r_inner_target_midpoints.putVar(&r_inner_target_midpoints[0]);
      _z_inner_target_midpoints.putVar(&z_inner_target_midpoints[0]);
      _rmrs_inner_target.putVar(&rmrs_inner_target[0]);
      _rmrs_inner_target_midpoints.putVar(&rmrs_inner_target_midpoints[0]);
      _Bmag_inner_target.putVar(&Bmag_inner_target[0]);
      _Bangle_inner_target.putVar(&Bangle_inner_target[0]);
      _te_inner_target.putVar(&te_inner_target[0]);
      _ti_inner_target.putVar(&ti_inner_target[0]);
      _ne_inner_target.putVar(&ne_inner_target[0]);
      _flux_inner_target.putVar(&flux_inner_target[0]);
      _ni_inner_target.putVar(&ni_inner_target[0]);

      _r_outer_target.putVar(&r_outer_target[0]);
      _z_outer_target.putVar(&z_outer_target[0]);
      _r_outer_target_midpoints.putVar(&r_outer_target_midpoints[0]);
      _z_outer_target_midpoints.putVar(&z_outer_target_midpoints[0]);
      _rmrs_outer_target.putVar(&rmrs_outer_target[0]);
      _rmrs_outer_target_midpoints.putVar(&rmrs_outer_target_midpoints[0]);
      _Bmag_outer_target.putVar(&Bmag_outer_target[0]);
      _Bangle_outer_target.putVar(&Bangle_outer_target[0]);
      _te_outer_target.putVar(&te_outer_target[0]);
      _ti_outer_target.putVar(&ti_outer_target[0]);
      _ne_outer_target.putVar(&ne_outer_target[0]);
      _flux_outer_target.putVar(&flux_outer_target[0]);
      _ni_outer_target.putVar(&ni_outer_target[0]);

      _gridr.putVar(&r_h[0]);
      _gridz.putVar(&z_h[0]);
      _vals.putVar(&val_h[0]);
      _te.putVar(solps_fields->te.data());
      _ti.putVar(solps_fields->ti.data());
      _ne.putVar(solps_fields->ne.data());
      _ni.putVar(solps_fields->ni.data());
      //_flux.putVar(solps_fields->flux_last.data());
      _mass.putVar(solps_fields->mass.data());
      _charge.putVar(solps_fields->charge.data());
      _Br.putVar(solps_fields->Br.data());
      _Bt.putVar(solps_fields->Bt.data());
      _Bz.putVar(solps_fields->Bz.data());
      _Bmag.putVar(solps_fields->Bmag.data());
      _vr.putVar(solps_fields->vr.data());
      _vt.putVar(solps_fields->vt.data());
      _vz.putVar(solps_fields->vz.data());
      _Er.putVar(solps_fields->Er.data());
      _Ez.putVar(solps_fields->Ez.data());
      _Et.putVar(&Et[0]);
      _gradTe.putVar(solps_fields->gradTe.data());
      _gradTer.putVar(solps_fields->gradTer.data());
      _gradTet.putVar(solps_fields->gradTet.data());
      _gradTez.putVar(solps_fields->gradTez.data());
      _gradTi.putVar(solps_fields->gradTi.data());
      _gradTir.putVar(solps_fields->gradTir.data());
      _gradTit.putVar(solps_fields->gradTit.data());
      _gradTiz.putVar(solps_fields->gradTiz.data());
      ncFile_out.close();
    
    auto end_time_clock = app_time::now();
    fsec out_time = end_time_clock - run_time_clock;
    printf("Time taken for output is %6.3f (secs) \n", out_time.count());
    return 0;
}
