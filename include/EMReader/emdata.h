#pragma once

#define H5_USE_16_API
#ifndef MAXPATHLEN
#define MAXPATHLEN 1024
#endif

#include <cmath>
#include <iostream>
#include <string>

#include "constants.h"
#include "emhdf.h"
#include "emhdf2.h"
#include "type.h"
#include "utils.h"

class Euler 
{
    public:
    Euler();
    ~Euler();
    Euler(float a1, float a2, float a3);                     // EMAN by default
    Euler(float a1, float a2, float a3, int typ);            // EMAN, IMAGIC or SPIDER
    Euler(float e1, float e2, float e3, float e0, int typ);  // QUATERNIONS, SPIN axis, or SGIROT
    Euler(float* m);                                         // matrix form
    enum EulerType { EMAN, IMAGIC, SPIN, QUAT, MATRIX, SGIROT, SPIDER, MRC };

    void setAngle(float a1, float a2, float a3, int typ = EMAN);  //  EMAN, IMAGIC or SPIDER setAngle
    void setAngle(float e1, float e2, float e3, float e0, int typ);   // quaternion, spin, and SGIROT setAngle
    void setAngle(float* m);  // matrix setAngle
    void rectify();
    float alt() const { return alt_; }
    float az() const { return az_; }
    float phi() const { return phi_; }

    private:
    int typ_; // The representation of the orientation that the user is employing
    float alt_, az_, phi_; // EMAN and internal variables
    float alt_mrc, az_mrc, phi_mrc; // mrc euler angle, before using them, need to call convertToMRCAngle()
    float m_[9];
    char sym[8];
    int nsym; // number of symmetric elements
    int csym; // current symmetric element to return
    void init();
};

struct emdataHeader 
{
    int samplebits;
    int flags;
    int rocount;
    float alipeak;
    int nx, ny, nz; // image size
    float dx, dy, dz; // translation from original image
    float alt, az, phi; // orientation of slice or model
    int nimg; // when averaging, # images in average

    float min, max; // image density limits
    int minloc, maxloc; // location of min and max value
    double mean, sigma; // image stats
    double sumsq;
    float mean2, sigma2; // these are statistics for a 4 pixel strip at the edge of the image

    float pixel; // A/pixel (or voxel)
    float xorigin, yorigin, zorigin; // the origin coordinates in Angstrom for the first pixel (0,0,0).

    char path[MAXPATHLEN]; // path image was read from
    int pathnum; // image number in file pointed to by path
    char name[80]; // descriptive name

    float ctf[11]; // ctf parameters, check HASCTF flag before using
                  // 0 - defocus (neg underfocus)
                  // 1 - B factor (A^2)
                  // 2 - amplitude
                  // 3 - ampcont (0 - 1)
                  // 4 - noise1	(noise3*exp(-((pi/2*noise4*x0)^2)-x0*noise2-sqrt(fabs(x0))*noise1))
                  // 5 - noise2	(noise expression is intensity)
                  // 6 - noise3
                  // 7 - noise4
                  // 8 - voltage (kv)
                  // 9 - Cs (mm)
                  // 10 - A/pix
    char micrograph_id[32];
    float particle_location_center_x;
    float particle_location_center_y;
    float center_x;
    float center_y;
    MapInfoType map_type;
    Euler::EulerType orientation_convention;
};

class emdata 
{
    char ptype[4]; // type of parameter sets
    float param[4][25]; // parameters
    int nimg;
    float * rdata;
    float * sup;
    int flags;
    long atime; // access time
    int tmp; // # of temp file when swapped out
    int nx, ny, nz;
    float dx, dy, dz;
    mrcH * mrch; // MRC header when read from MRC file
    float pixel;
    imagicH * imagich; // imagic header when read from imagic file
    SPIDERH * spiderH; // spider header when read from spider file
    float xorigin, yorigin, zorigin;
    emdata * parent; // if this image was generated from another, this is it
    emdata * fft; // this avoids doing multiple ffts of the same set
    emdata * rfp; // avoids multiple rotational footprints
    float * frm2dhhat; // avoids multiple FRM2D H_hat
    float particle_location_center_x;
    float particle_location_center_y;
    double mean, sigma;
    float min, max;
    float alipeak;
    float ctf[11];
    MapInfoType map_type;
    float center_x;
    float center_y;
    // static List *ralfp; // rotational alignment basis images
    static int rali, ralo, rals; // rotational alignment inner/outer rad and im size
    Euler::EulerType orientation_convention;
    Euler euler;

    public:
    emdataHeader header;
    int readImage(const char * filespec, int n, int nodata = 0);
    int readHDF(const char *, int, int);
    int readHDF2(const char *, int, int);
    int readMRC(const char *, int, int);
    emdata();
    ~emdata();
    int setSize(int x, int y = 1, int z = 1);
    void zero(float sig = 0);
    void ap2ri();
    void rotate(float e);
    inline float * getData() { return rdata; }
    inline void setXYZOrigin(float x, float y, float z) 
    {
        header.xorigin = x;
        header.yorigin = y;
        header.zorigin = z;
    };
    emdata * project3d(emdata * proj, float alt, float az, float phi, int mode, float thr = 1.0, int ri = 0, int dcx = 0, int dcy = 0, int dcz = 0);
    inline int xSize() { return header.nx; }
    inline int ySize() { return header.ny; }
    inline int zSize() { return header.nz; }
    inline void setMin(float m) { header.min = m; }
    inline void setMax(float m) { header.max = m; }
    inline void setMinLoc(int m) { header.minloc = m; }
    inline void setMaxLoc(int m) { header.maxloc = m; }
    inline void setMean(float m) { header.mean = m; }
    inline void setSigma(float s) { header.sigma = s; }
    inline void setSumSq(float ss) { header.sumsq = ss; }
    inline void setMean2(float m2) { header.mean2 = m2; }
    inline void setSigma2(float s2) { header.sigma2 = s2; }
    inline float Dx() { return header.dx; }
    inline float Dy() { return header.dy; }
    inline float Dz() { return header.dz; }
    int writeImage(const char * filename, int n, int headonly = false);
    int writeHDF2(const char * filename, int n, int headonly = 0);
    inline void setNImg(int n) { nimg = n; }
    inline float hypot3(float x, float y, float z) { return x * x + y * y + z * z; }
    inline float hypot2(float x, float y) { return x * x + y * y; }
    inline int floor2(float x) 
    {
        if (x < 0) return (int)x - 1;
        return (int)x;
    }
    inline float trilin(float p1, float p2, float p3, float p4, float p5, float p6, float p7, float p8, float t, float u, float v) 
    {
        return (1.0 - t) * (1.0 - u) * (1.0 - v) * p1 + t * (1.0 - u) * (1.0 - v) * p2 + (1.0 - t) * u * (1.0 - v) * p3 + t * u * (1.0 - v) * p4 + (1.0 - t) * (1.0 - u) * v * p5 + t * (1.0 - u) * v * p6 + (1.0 - t) * u * v * p7 + t * u * v * p8;
    }
    void applyMask(int r, int type, float dx0 = 0, float dy0 = 0, float dz0 = 0);
    inline Euler * getEuler() { return &euler; }
    inline void setName(const char * n) 
    {
        std::strncpy(header.name, n, 79);
        header.name[79] = 0;
    }
    inline void setPath(const char * p) 
    {
        std::strncpy(header.path, p, MAXPATHLEN - 1);
        header.path[MAXPATHLEN - 1] = 0;
    }
    inline void setMicrographId(const char * theId) 
    {
        strncpy(header.micrograph_id, theId, 31);
        header.micrograph_id[31] = 0;
    }
    inline void setRAlign(float Alt, float Az, float Phi) 
    {
        if (!std::isfinite(Alt)) Alt = 0;
        if (!std::isfinite(Az)) Az = 0;
        if (!std::isfinite(Phi)) Phi = 0;
        euler.setAngle(Alt, Az, Phi);
    }
};

class EMObject 
{
    public:
    enum ObjectType { INT, FLOAT, DOUBLE, STRING, EMDATA, XYDATA, FLOATARRAY, STRINGARRAY, UNKNOWN };

    public:
    EMObject() : type(UNKNOWN) 
    {
        n = 0;
        f = 0;
        d = 0;
        emobj = 0;
    }

    EMObject(int num) : n(num), emobj(0), type(INT) {}
    EMObject(float ff) : f(ff), emobj(0), type(FLOAT) {}
    EMObject(double dd) : d(dd), emobj(0), type(DOUBLE) {}
    EMObject(const char * s) : n(0), emobj(0), str(std::string(s)), type(STRING) {}
    EMObject(const std::string & s) : n(0), emobj(0), str(s), type(STRING) {}
    EMObject(const std::vector<float> & v) : n(0), emobj(0), farray(v), type(FLOATARRAY) {}
    EMObject(const std::vector<std::string>& sarray)
      : n(0), emobj(0), strarray(sarray), type(STRINGARRAY) {}

    ~EMObject() {}

    ObjectType get_type() const { return type; }
    operator int() const;
    operator float() const;
    operator double() const;
    operator const char*() const;
    operator std::vector<float>() const;
    operator std::vector<std::string>() const;

    int n;
    float f;
    double d;
    emdata* emobj;
    std::string str;
    ObjectType type;
    std::vector<float> farray;
    std::vector<std::string> strarray;
};