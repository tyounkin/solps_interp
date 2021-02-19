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

    saxpy_functor(int* _index_triangle, int _nr, int _nz,double* _r, double* _z,
        double* _r1, double* _z1, double* _v1,
        double* _r2, double* _z2, double* _v2,
        double* _r3, double* _z3, double* _v3,
        double* _radius, double* _val, bool* _found) : 
      index_triangle(_index_triangle), nr(_nr), nz(_nz), r(_r), z(_z),
      r1(_r1), z1(_z1), v1(_v1),
      r2(_r2), z2(_z2), v2(_v2),
      r3(_r3), z3(_z3), v3(_v3),
      radius (_radius), val(_val), found(_found) {}

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
               }
            }
        }
};

// Global variables

int nx, ny;

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
           std::vector<double>,std::vector<double>,
           std::vector<double>,std::vector<double>,
           std::vector<double>,std::vector<double>>
            get_fields_triangles(std::vector<double> Er, std::vector<double> Ez,
           std::vector<double> r1,std::vector<double> r2,
           std::vector<double> r3,std::vector<double> z1,
           std::vector<double> z2,std::vector<double> z3,
           std::vector<double> v1,std::vector<double> v2,
           std::vector<double> v3,std::vector<double> radius)
{
  std::vector<int> nxny = read_ifield("b2fgmtry","nx,ny");

  nx = nxny[0];
  ny = nxny[1];

  // Calculate Efield on SOLPS grid
  Er.resize((nx+2)*(ny+2),0.0);
  Ez.resize((nx+2)*(ny+2),0.0);
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
    }
  }

  return std::make_tuple(Er, Ez, r1, r2, r3, z1, z2, z3, v1, v2, v3, radius);
}
int main()
{
    typedef std::chrono::high_resolution_clock app_time;
    auto app_start_time = app_time::now();
    
    std::cout << "Hello World!\n";
    
    //try {
    //  netCDF::NcFile ncp0("../solps_triangles.nc", netCDF::NcFile::read);
    //}
    //catch (netCDF::exceptions::NcException &e) {
    //  e.what();
    //  std::cout << "FAILURE to open file" << std::endl;
    //  return -1;
    //}

    int nx = 90;
    int ny = 36;
    int n8 = 8;
    int n_total = 0;

    
    //netCDF::NcFile ncp("../solps_triangles.nc", netCDF::NcFile::read);
    //netCDF::NcDim _nx(ncp.getDim("nx"));
    //netCDF::NcDim _ny(ncp.getDim("ny"));
    //
    //nx = _nx.getSize();
    //ny = _ny.getSize();

    n_total = nx*ny*n8;
    ////std::vector<double> r1(n_total), r2(n_total), r3(n_total),
    ////                    z1(n_total), z2(n_total), z3(n_total),
    ////                    v1(n_total), v2(n_total), v3(n_total),
    ////                    radius(n_total);
    //
    thrust::host_vector<double> r1_h(n_total), r2_h(n_total), r3_h(n_total),
                        z1_h(n_total), z2_h(n_total), z3_h(n_total),
                        v1_h(n_total), v2_h(n_total), v3_h(n_total),
                        radius_h(n_total);

    //netCDF::NcVar ncp_r1(ncp.getVar("r1"));
    //netCDF::NcVar ncp_r2(ncp.getVar("r2"));
    //netCDF::NcVar ncp_r3(ncp.getVar("r3"));
    //netCDF::NcVar ncp_z1(ncp.getVar("z1"));
    //netCDF::NcVar ncp_z2(ncp.getVar("z2"));
    //netCDF::NcVar ncp_z3(ncp.getVar("z3"));
    //netCDF::NcVar ncp_v1(ncp.getVar("v1"));
    //netCDF::NcVar ncp_v2(ncp.getVar("v2"));
    //netCDF::NcVar ncp_v3(ncp.getVar("v3"));
    //netCDF::NcVar ncp_rad(ncp.getVar("radius"));
    //
    //ncp_r1.getVar(&r1_h[0]);
    //ncp_r2.getVar(&r2_h[0]);
    //ncp_r3.getVar(&r3_h[0]);
    //ncp_z1.getVar(&z1_h[0]);
    //ncp_z2.getVar(&z2_h[0]);
    //ncp_z3.getVar(&z3_h[0]);
    //ncp_v1.getVar(&v1_h[0]);
    //ncp_v2.getVar(&v2_h[0]);
    //ncp_v3.getVar(&v3_h[0]);
    //ncp_rad.getVar(&radius_h[0]);
    //ncp.close();
// Calculate Efield on SOLPS grid
  std::vector<double> Er, Ez;
  std::vector<double> r1t, r2t, r3t, z1t, z2t, z3t, v1t, v2t, v3t, radiust;

  std::tie(Er, Ez, r1t, r2t, r3t, z1t, z2t, z3t, v1t, v2t, v3t, radiust) = get_fields_triangles(Er, Ez, r1t, r2t, r3t, z1t, z2t, z3t, v1t, v2t, v3t, radiust);
   
      netCDF::NcFile ncFile_tri("solps_triangles2.nc",
                         netCDF::NcFile::replace);
      netCDF::NcDim _nx = ncFile_tri.addDim("nx", nx);
      netCDF::NcDim _ny = ncFile_tri.addDim("ny", ny);
      netCDF::NcDim _n8 = ncFile_tri.addDim("n8", n8);
      std::vector<netCDF::NcDim> outdimt;
      outdimt.push_back(_n8);
      outdimt.push_back(_ny);
      outdimt.push_back(_nx);

      netCDF::NcVar _r1 = ncFile_tri.addVar("r1", netCDF::ncDouble, outdimt);
      netCDF::NcVar _r2 = ncFile_tri.addVar("r2", netCDF::ncDouble, outdimt);
      netCDF::NcVar _r3 = ncFile_tri.addVar("r3", netCDF::ncDouble, outdimt);
      netCDF::NcVar _z1 = ncFile_tri.addVar("z1", netCDF::ncDouble, outdimt);
      netCDF::NcVar _z2 = ncFile_tri.addVar("z2", netCDF::ncDouble, outdimt);
      netCDF::NcVar _z3 = ncFile_tri.addVar("z3", netCDF::ncDouble, outdimt);
      netCDF::NcVar _v1 = ncFile_tri.addVar("v1", netCDF::ncDouble, outdimt);
      netCDF::NcVar _v2 = ncFile_tri.addVar("v2", netCDF::ncDouble, outdimt);
      netCDF::NcVar _v3 = ncFile_tri.addVar("v3", netCDF::ncDouble, outdimt);
      netCDF::NcVar _radius = ncFile_tri.addVar("radius", netCDF::ncDouble, outdimt);
      _r1.putVar(&r1t[0]);
      _r2.putVar(&r2t[0]);
      _r3.putVar(&r3t[0]);

      _z1.putVar(&z1t[0]);
      _z2.putVar(&z2t[0]);
      _z3.putVar(&z3t[0]);
      _v1.putVar(&v1t[0]);
      _v2.putVar(&v2t[0]);
      _v3.putVar(&v3t[0]);
      _radius.putVar(&radiust[0]);
      ncFile_tri.close();
    //thrust::copy(r1_h.begin(), r1_h.end(), r1t.begin()); 
    //thrust::copy(r2_h.begin(), r2_h.end(), r2t.begin()); 
    //thrust::copy(r3_h.begin(), r3_h.end(), r3t.begin()); 
    //thrust::copy(z1_h.begin(), z1_h.end(), z1t.begin()); 
    //thrust::copy(z2_h.begin(), z2_h.end(), z2t.begin()); 
    //thrust::copy(z3_h.begin(), z3_h.end(), z3t.begin()); 
    //thrust::copy(v1_h.begin(), v1_h.end(), v1t.begin()); 
    //thrust::copy(v2_h.begin(), v2_h.end(), v2t.begin()); 
    //thrust::copy(v3_h.begin(), v3_h.end(), v3t.begin()); 
    //thrust::copy(radius_h.begin(), radius_h.end(), radiust.begin()); 
    //thrust::device_vector<double> r1 = r1_h;
    //thrust::device_vector<double> r2 = r2_h;
    //thrust::device_vector<double> r3 = r3_h;
    //thrust::device_vector<double> z1 = z1_h;
    //thrust::device_vector<double> z2 = z2_h;
    //thrust::device_vector<double> z3 = z3_h;
    //thrust::device_vector<double> v1 = v1_h;
    //thrust::device_vector<double> v2 = v2_h;
    //thrust::device_vector<double> v3 = v3_h;
    //thrust::device_vector<double> radius = radius_h;
    thrust::device_vector<double> r1(r1t);
    thrust::device_vector<double> r2(r2t);
    thrust::device_vector<double> r3(r3t);
    thrust::device_vector<double> z1(z1t);
    thrust::device_vector<double> z2(z2t);
    thrust::device_vector<double> z3(z3t);
    thrust::device_vector<double> v1(v1t);
    thrust::device_vector<double> v2(v2t);
    thrust::device_vector<double> v3(v3t);
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
    thrust::counting_iterator<std::size_t> point_first(0);
    thrust::counting_iterator<std::size_t> point_last(nr*nz);
    double r_start = 4.0;
    double r_end = 8.4;
    double z_start = -4.6;
    double z_end = 4.7;

    double dr = (r_end - r_start)/(nr - 1);
    double dz = (z_end - z_start)/(nz - 1);

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
                      radius_pointer,val_pointer,found_pointer);
    
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
      std::vector<netCDF::NcDim> outdim;
      outdim.push_back(_nz);
      outdim.push_back(_nr);

      netCDF::NcVar _gridr = ncFile_out.addVar("gridr", netCDF::ncDouble, _nr);
      netCDF::NcVar _gridz = ncFile_out.addVar("gridz", netCDF::ncDouble, _nz);
      netCDF::NcVar _vals = ncFile_out.addVar("values", netCDF::ncDouble, outdim);

      _gridr.putVar(&r_h[0]);
      _gridz.putVar(&z_h[0]);
      _vals.putVar(&val_h[0]);
      ncFile_out.close();
    
    auto end_time_clock = app_time::now();
    fsec out_time = end_time_clock - run_time_clock;
    printf("Time taken for output is %6.3f (secs) \n", out_time.count());
    return 0;
}
