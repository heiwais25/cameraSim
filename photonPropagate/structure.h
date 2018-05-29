#ifndef _STRUCTURE_
#define _STRUCTURE_

#include<vector>
#include<map>
#include<iostream>
#include<cstdlib>
#include"define.h"
using namespace std;





struct DOM{
  float r[3];
};

// Declare the DOM uniquness
struct ikey{
    int str, dom;

    bool isInIce() const{
        return str>0 && dom>=1 && dom<=60;
    }

    bool operator< (const ikey & rhs) const {
        return str == rhs.str ? dom < rhs.dom : str < rhs.str;
    }

    bool operator!= (const ikey & rhs) const {
        return str != rhs.str || dom != rhs.dom;
    }
};

// DOM individual info
struct name:ikey{
    int type;
    float rde, hv; // DOM's electrical property

    name(){}
    name(ikey k, float r, int t, float h):ikey(k){
        rde = r; type = t; hv = h;
    }
};

struct hit{
	unsigned int i;
	float t;
	unsigned int n;
	float z;
#ifdef PDIR
	float pth, pph, dth, dph;
#endif
};

struct pbuf{
	float4 r;        // location, time
	float3 n;        // direction
	unsigned int q;  // track segment
};

struct photon:pbuf{
  float l;         // track length
#ifdef ANGW
  float f;         // fraction of light from muon alone (without cascades)
#endif
#ifdef LONG
  float a, b;      // longitudinal development parametrization coefficients
#endif
};

struct ices{
  float wvl;             // wavelength of this block
  float ocm;             // 1 / speed of light in medium
  float coschr, sinchr;  // cos and sin of the cherenkov angle
  struct{
    float abs;           // absorption
    float sca;           // scattering
  } z [MAXLYS];
};

struct line{
  short n, max; 	// n : fisrt DOM index in string, max : total number of OM in string
  float x, y, r;	// x, y : average position of string, r : maximum radius of DOM in string
  float h, d; 		// h : highest OM z-axis position in string, d : OM density(number of OM per distance)
  float dl, dh;		// dl, dh : low and high error bound of OM for geometry(z-axis position for each OM)
};


struct datz{
  ices w[WNUM];
  unsigned int rm[MAXRND];
  unsigned long long rs[MAXRND];
};

struct dats{
  unsigned int hidx; 	//

#ifndef XCPU
  unsigned int tn, tx;  // kernel time clocks
  unsigned int ab;      // if TOT was abnormal
  unsigned int mp;      // kernel block counter
  short bmp[4];         // list of 4 faulty MPs
#endif
  short blockIdx, gridDim;  // bad/current MP; number of MPs

  int type;   // 0=cascade/1=flasher/2=flasher 45/3=laser up/4=laser down
  float r[3]; // flasher/laser coordinates
  float ka, up;    // 2d-gaussian rms and zenith of cone

  unsigned int hnum;    // maximum size of hits buffer
  int size;   // size of kurt table
  int rsize;  // count of multipliers
  int gsize;  // count of initialized OMs

  float dh, hdh, rdh, hmin; // step, step/2, 1/step, and min depth

  float ocv;  // 1 / speed of light in vacuum
  float sf;   // scattering function: 0=HG; 1=SAM
  float g, g2, gr; // g=<cos(scattering angle)>, g2=g*g and gr=(1-g)/(1+g)
  float R, R2, zR; // DOM radius, radius^2, and inverse "oversize" scaling factor

#ifdef HOLE
  float hr, hr2, hs, ha; // hole ice radius, radius^2, effective scattering and absorption coefficients
  float SF, G, G2, GR;   // hole ice sf, g, g2, gr
#endif

  int cn[2]; // 
  float cl[2], crst[2]; // cl : minimum position of each direction, crst : interval / distance = (density?)

  unsigned char is[CX][CY]; // 
  unsigned char ls[NSTR];   // 
  line sc[NSTR]; // information of each string
  float rx; // maximum radius among strings

  float fldr; // horizontal direction of the flasher led #1
  float eff;  // OM efficiency correction

#ifdef ASENS
  float mas;  // maximum angular sensitivity
  float s[ANUM]; // ang. sens. coefficients
#endif

#ifdef ROMB
  float cb[2][2];
#endif
#ifdef TILT
  int lnum, lpts, l0;
  float lmin, lrdz, r0;
  float lnx, lny;
  float lr[LMAX];
  float lp[LMAX][LYRS];
#endif

  short fla;
#ifdef ANIZ
  short k;          // ice anisotropy: 0: no, 1: yes
  float k1, k2, kz; // ice anisotropy parameters
  float azx, azy;   // ice anisotropy direction
#endif

  datz * z;
  hit * hits;
  photon * pz;
#ifdef TALL
  pbuf * bf;
#endif
};

struct doms{
  DOM oms[MAXGEO];
  name names[MAXGEO];
  map<float, float> rde;

  hit * hits;
  photon * pz;
};

#ifdef XCPU
struct float2{
  float x, y;
};

struct float3:float2{
  float z;
};

struct float4:float3{
  float w;
};
#endif





#endif