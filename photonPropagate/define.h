#ifndef _DEFINE_
#define _DEFINE_

#define OFLA         // omit the flasher DOM
// #define ROMB         // use rhomb cells aligned with the array
#define ASENS        // enable angular sensitivity
#define RAND         // disable for deterministic results
#define CAMERA
// #define TILT         // enable tilted ice layers
#define ANIZ         // enable anisotropic ice
#define HOLE         // enable direct hole ice simulation
#define PDIR       // record photon arrival direction and impact point

#define MKOW         // photon yield parametrizations by M. Kowalski
#define ANGW         // smear cherenkov cone due to shower development
#define LONG         // simulate longitudinal cascade development
#define CWLR         // new parameterizations by C. Wiebusch and L. Raedel
                     // requires that MKOW, ANGW, and LONG are all defined

#ifdef ASENS
#define ANUM 11      // number of coefficients in the angular sensitivity curve
#endif

// For SpiceHD simulation enable ASENS and enable HOLE
// (tracking through the hole ice column and only accepting
// photons on the lower hemisphere, where the PMT is situated)
// Be sure to use an as.dat with two lines: 0.0 and 0.33

#ifdef TILT
#define LMAX 6       // number of dust loggers
#define LYRS 170     // number of depth points
#endif

#ifdef ROMB
#define DIR1 9.3
#define DIR2 129.3

#define CX   21
#define CY   19
#define NSTR 94
#else
#define CX   2  // Originally, it is 13. But we need only 2 string in one direction
#define CY   1
#define NSTR 94
#endif

#ifdef XCPU
#define OVER 1
#define NBLK 1
#define NTHR 512
#else
#define OVER 10      // size of photon bunches along the muon track
#endif

#define TALL         // enable faster 2-stage processing, takes more memory
#define HQUO   1     // save at most photons/HQUO hits
#define NPHO   1024  // maximum number of photons propagated by one thread

#define WNUM   32    // number of wavelength slices
#define MAXLYS 180   // maximum number of ice layers
#define MAXGEO 5200  // maximum number of OMs
#define MAXRND 131072   // max. number of random number multipliers

#define XXX 1.e-5f
#define FPI 3.141592653589793f
#define OMR 0.16510f // DOM radius [m]

#endif