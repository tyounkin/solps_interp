#include <iostream>
#include <netcdf>
#include <vector>
#include <cmath>
#include <omp.h>

void vectorPrint(std::string a, double D[])
{
    std::cout << "vector " << a << " " << 
    D[0] << " " <<  
    D[1] << " " << 
    D[2] << std::endl;
}
void vectorAssign(double a, double b,double c, double D[])
{
    D[0] = a;
    D[1] = b;
    D[2] = c;
}

void vectorSubtract(double A[], double B[],double C[])
{
    C[0] = A[0] - B[0];
    C[1] = A[1] - B[1];
    C[2] = A[2] - B[2];
}

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

double vectorDotProduct(double A[], double B[])
{
    double c = A[0]*B[0] +  A[1]*B[1] + A[2]*B[2];
    return c;
}

double vectorNorm(double A[])
{
    double norm = 0.0f;
    norm = std::sqrt(A[0]*A[0] + A[1]*A[1] + A[2]*A[2]);

        return norm;
}
double interpolate_value(double r1, double z1, double a1, double r2, double z2, double a2, double r3, double z3, double a3, double r_point, double z_point)
{
                    double denom = (z2 - z3)*(r1 - r3) + (r3 - r2)*(z1 - z3);
                    double weights0 = ((z2 - z3)*(r_point - r3) + (r3 - r2)*(z_point - z3))/denom;
                    double weights1 = ((z3 - z1)*(r_point - r3) + (r1 - r3)*(z_point - z3))/denom;
                    double weights2 = 1.0 - weights0 - weights1;
                    double a_point = a1*weights0 + a2*weights1 + a3*weights2;

                    return a_point;
}
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

int main()
{
    std::cout << "Hello World!\n";
    
    try {
      netCDF::NcFile ncp0("../solps_triangles.nc", netCDF::NcFile::read);
    }
    catch (netCDF::exceptions::NcException &e) {
      e.what();
      std::cout << "FAILURE to open file" << std::endl;
      return -1;
    }

    int nx = 0;
    int ny = 0;
    int n8 = 8;
    int n_total = 0;

    
    netCDF::NcFile ncp("../solps_triangles.nc", netCDF::NcFile::read);
    netCDF::NcDim _nx(ncp.getDim("nx"));
    netCDF::NcDim _ny(ncp.getDim("ny"));
    
    nx = _nx.getSize();
    ny = _ny.getSize();

    n_total = nx*ny*n8;

    std::vector<double> r1(n_total), r2(n_total), r3(n_total),
                        z1(n_total), z2(n_total), z3(n_total),
                        v1(n_total), v2(n_total), v3(n_total),
                        radius(n_total);

    netCDF::NcVar ncp_r1(ncp.getVar("r1"));
    netCDF::NcVar ncp_r2(ncp.getVar("r2"));
    netCDF::NcVar ncp_r3(ncp.getVar("r3"));
    netCDF::NcVar ncp_z1(ncp.getVar("z1"));
    netCDF::NcVar ncp_z2(ncp.getVar("z2"));
    netCDF::NcVar ncp_z3(ncp.getVar("z3"));
    netCDF::NcVar ncp_v1(ncp.getVar("v1"));
    netCDF::NcVar ncp_v2(ncp.getVar("v2"));
    netCDF::NcVar ncp_v3(ncp.getVar("v3"));
    netCDF::NcVar ncp_rad(ncp.getVar("radius"));
    
    ncp_r1.getVar(&r1[0]);
    ncp_r2.getVar(&r2[0]);
    ncp_r3.getVar(&r3[0]);
    ncp_z1.getVar(&z1[0]);
    ncp_z2.getVar(&z2[0]);
    ncp_z3.getVar(&z3[0]);
    ncp_v1.getVar(&v1[0]);
    ncp_v2.getVar(&v2[0]);
    ncp_v3.getVar(&v3[0]);
    ncp_rad.getVar(&radius[0]);
    ncp.close();
 
    int nr = 440;
    int nz = 930;
    double r_start = 4.0;
    double r_end = 8.4;
    double z_start = -4.6;
    double z_end = 4.7;

    double dr = (r_end - r_start)/(nr - 1);
    double dz = (z_end - z_start)/(nz - 1);

    std::vector<double> r(nr), z(nz), val(nr*nz);

    for(int i=0; i< nr; i++)
    {
      r[i] = r_start + i*dr;
    }
    
    for(int j=0; j < nz; j++)
    {
      z[j] = z_start + j*dz;
    }
    
    //int nr = 1;
    //int nz = 1;
    //std::vector<double> r(1,6.0);
    //std::vector<double> z(1,0.0);
    //std::vector<double> val(1,0.0);
    double hollow_r0 = 4.65;
    double hollow_r1 = 7.3;
    double hollow_z0 = -0.7;
    double hollow_z1 = 2.35;
#pragma omp parallel for collapse(3)
    for(int i=0; i< n_total; i++)
    {
        for(int jj=0; jj < nz; jj++)
        {
          for(int ii=0; ii< nr; ii++)
          {
            //if((r[ii] > hollow_r0) && (r[ii] < hollow_r1) && (z[jj] > hollow_z0) && (z[jj] < hollow_z1) )
            //{
            //}
            //else
            //{
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
                  val[jj*nr + ii] = a_point;
                }
              }
            //}
          }
        }
      
    }

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

      _gridr.putVar(&r[0]);
      _gridz.putVar(&z[0]);
      _vals.putVar(&val[0]);
      ncFile_out.close();
    return 0;
}
