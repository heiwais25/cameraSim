#include <map>
#include <deque>
#include <vector>
#include <cmath>
#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <sys/time.h>

#ifndef __CUDACC__
#define XCPU
#elif __CUDA_ARCH__ >= 120
#define USMA
#endif

#ifdef XCPU
#include <cmath>
#include <cstring>
#endif

using namespace std;

namespace xppc{
	#include "ini.cxx"
	#include "pro.cu"
	
	void initialize(float enh = 1.f){ 
		m.set(); 
		d.eff *= enh; 
	}
	
	unsigned int pmax, pmxo, pn;
	
	#ifdef XCPU
	dats *e;  // pointer to a copy of "d" on device
	int numBlock, threadPerBlock, ntot;
	
	void ini(int type){
		rs_ini();
		pn=0;
		
		ntot=numBlock*threadPerBlock;
		pmax=ntot*NPHO;
		pmxo=pmax/OVER;
		pmax=pmxo*OVER;
		d.hnum=pmax/HQUO;
		
		{
			d.hits = q.hits = new hit[d.hnum];
			if(type==0) d.pz = q.pz = new photon[pmxo];
			#ifdef TALL
			d.bf = new pbuf[pmax];
			#endif
		}
		
		{
			d.z=&z; e=&d; oms=q.oms;
		}
		
		{
			unsigned int size=d.rsize, need=seed+1;
			if(size<need) cerr<<"Error: not enough multipliers: asked for "<<seed<<"-th out of "<<size<<"!"<<endl;
		}
	}
	
	void fin(){
		if(d.type==0) delete d.pz;
		delete d.hits;
		#ifdef TALL
		delete d.bf;
		#endif

	}
	#else
	bool xgpu=false;
	
	void checkError(cudaError result){
		if(result!=cudaSuccess){
			cerr<<"CUDA Error: "<<cudaGetErrorString(result)<<endl;
			exit(2);
		}
	}
	
	struct gpu{
		dats d;
		dats *e;  // pointer to a copy of "d" on device
		
		int device;
		int numBlock, threadPerBlock, ntot; // total threads in parallel
		unsigned int npho, pmax, pmxo;
		
		float dt, deviceTime, threadMin, threadMax;
		cudaDeviceProp prop;
		cudaStream_t stream;
		cudaEvent_t evt1, evt2;
		
		unsigned int old, num;
		
		gpu(int device) : deviceTime(0), threadMin(0), threadMax(0), old(0), npho(NPHO){
			this->device=device;
			
			{
				ostringstream o; o<<"NPHO_"<<device;
				char * nph=getenv(o.str().c_str());
				if(nph==NULL) nph=getenv("NPHO");
				if(nph!=NULL) if(*nph!=0){
					npho=atoi(nph);
					cerr<<"Setting NPHO="<<npho<<endl;
					if(npho<=0){
						cerr<<"Not using device # "<<device<<"!"<<endl;
						return;
					}
				}
			}
			
			checkError(cudaSetDevice(device));
			checkError(cudaGetDeviceProperties(&prop, device));
			
			#if CUDART_VERSION >= 3000
			checkError(cudaFuncSetCacheConfig(propagate, cudaFuncCachePreferL1));
			#endif
			
			cudaFuncAttributes attr;
			checkError(cudaFuncGetAttributes (&attr, propagate));
			
			numBlock = prop.multiProcessorCount;
			threadPerBlock = attr.maxThreadsPerBlock;
			// threadPerBlock = 512;

			// Change copy process?
			// numBlock = prop.multiProcessorCount * 2;
			// threadPerBlock = 256;

			cerr << "Running on " << numBlock << " blocks x " << threadPerBlock << " threads" << endl;
			
			fprintf(stderr, "Kernel uses: l=%lu r=%d s=%lu c=%lu\n", (unsigned long)attr.localSizeBytes,
			attr.numRegs, (unsigned long)attr.sharedSizeBytes, (unsigned long)attr.constSizeBytes);
		}
		
		void ini(int type){
			// init the random number generator
			rs_ini();
			d = xppc::d;
			
			// Check BADMP 
			{
				d.blockIdx = -1, d.gridDim=numBlock;
				ostringstream o; o<<"BADMP_"<<device;
				char * bmp=getenv(o.str().c_str());
				if(bmp==NULL) bmp=getenv("BADMP");
				if(bmp!=NULL) if(*bmp!=0){
					d.blockIdx=atoi(bmp), d.gridDim--;
					cerr<<"Not using MP #"<<d.blockIdx<<endl;
				}
			}
			
			ntot = numBlock * threadPerBlock;
			
			{
				unsigned long xmem = prop.totalGlobalMem;
				
				while(npho>1){
					pmax = ntot * npho;
					pmxo = pmax / OVER;
					pmax = pmxo*OVER;
					d.hnum = pmax/HQUO; // save at most photon
					
					unsigned long mtot = sizeof(datz) + sizeof(dats) + d.gsize * sizeof(DOM);
					mtot += +d.hnum*sizeof(hit);
					if(d.type==0) mtot += pmxo*sizeof(photon);
#ifdef TALL
					mtot += pmax*sizeof(pbuf);
#endif
					if(mtot > xmem) npho/=2; else break;
				}
			}
			
			{
				checkError(cudaStreamCreate(&stream));
				checkError(cudaEventCreateWithFlags(&evt1, cudaEventBlockingSync));
				checkError(cudaEventCreateWithFlags(&evt2, cudaEventBlockingSync));
			}
			
			{
				unsigned int size=d.rsize;
				if(size<ntot) cerr<<"Error: not enough multipliers: only have "<<size<<" (need "<<ntot<<")!"<<endl;
				else d.rsize=ntot;
			}
			
			unsigned long tot=0, cnt=0;
			
			{
				unsigned long size=sizeof(datz); tot+=size;
				checkError(cudaMalloc((void**) &d.z, size));
				checkError(cudaMemcpy(d.z, &z, size, cudaMemcpyHostToDevice));
			}
			
			{
				unsigned long size=d.hnum*sizeof(hit); tot+=size;
				checkError(cudaMalloc((void**) &d.hits, size));
			}
			
			if(d.type==0){
				unsigned long size=pmxo*sizeof(photon); tot+=size;
				checkError(cudaMalloc((void**) &d.pz, size));
			}
			
#ifdef TALL
			{
				unsigned long size=pmax*sizeof(pbuf); tot+=size;
				checkError(cudaMalloc((void**) &d.bf, size));
			}
#endif
			
			{
				unsigned long size=d.gsize*sizeof(DOM); cnt+=size;
				checkError(cudaMemcpyToSymbol(oms, q.oms, size));
			}
			
			{
				unsigned long size=sizeof(dats); tot+=size;
				checkError(cudaMalloc((void**) &e, size));
				checkError(cudaMemcpy(e, &d, size, cudaMemcpyHostToDevice));
			}
			
			cerr << "Total GPU memory usage: "<< tot << "  const: " << cnt << "  (npho="<<npho<<")"<<endl;
		}
		
		void fin(){
			checkError(cudaFree(d.z));
			checkError(cudaFree(d.hits));
			if(d.type==0) checkError(cudaFree(d.pz));
			#ifdef TALL
			checkError(cudaFree(d.bf));
			#endif
			checkError(cudaFree(e));
			checkError(cudaEventDestroy(evt1));
			checkError(cudaEventDestroy(evt2));
			checkError(cudaStreamDestroy(stream));

			// closeFile();
			
		}
		
		void set(){
			if(xgpu) checkError(cudaSetDevice(device));
		}
		
		void kernel_i(){
			{

				checkError(cudaStreamSynchronize(stream));
				checkError(cudaMemcpy(&d, e, 7*sizeof(int), cudaMemcpyDeviceToHost));
				
				checkError(cudaEventElapsedTime(&dt, evt1, evt2)); deviceTime+=dt;
				
				if(d.ab>0){
					cerr<<"Error: TOT was a nan or an inf "<<d.ab<<" times! Bad GPU "<<device<<" MP";
					for(int i=0; i<min(d.ab, 4); i++) cerr<<" #"<<d.bmp[i]; cerr<<endl;
				}
				if(d.mp!=d.gridDim){ cerr<<"Error: did not encounter MP #"<<d.blockIdx<<endl; exit(4); }
				if(threadMax!=-1){
					if((unsigned long long)(dt*prop.clockRate)<0x100000000ULL){
						threadMin+=d.tn/(float)prop.clockRate;
						threadMax+=d.tx/(float)prop.clockRate;
					}
					else threadMin=-1, threadMax=-1;
				}
				
				if(d.hidx>=d.hnum){ d.hidx=d.hnum; cerr<<"Error: data buffer overflow occurred!"<<endl; }
			}
			
			{
				unsigned int size=d.hidx*sizeof(hit);
				checkError(cudaMemcpyAsync(&q.hits[xppc::d.hidx], d.hits, size, cudaMemcpyDeviceToHost, stream));
				xppc::d.hidx+=d.hidx;
			}
		}
		
		void kernel_c(unsigned int & idx){
			if(old>0) checkError(cudaStreamSynchronize(stream));
			unsigned int pn=num/OVER;
			unsigned int size=pn*sizeof(photon);
			checkError(cudaMemcpyAsync(d.pz, &q.pz[idx], size, cudaMemcpyHostToDevice, stream));
			idx+=pn;
		}
		
		void kernel_f(){
			checkError(cudaStreamSynchronize(stream));
			if(num>0){
				checkError(cudaEventRecord(evt1, stream));
				
				propagate<<< 1, 1, 0, stream >>>(e, 0);
				checkError(cudaGetLastError());
				
				// propagate<<< numBlock, threadPerBlock, 0, stream >>>(e, num);
				propagate<<< numBlock, threadPerBlock, 0, stream >>>(e, num);
				checkError(cudaGetLastError());
				
				checkError(cudaEventRecord(evt2, stream));
				// checkError(cudaEventSynchronize(evt2));
			}
		}
		
		void stop(){
			fprintf(stderr, "Device time: %2.1f (in-kernel: %2.1f...%2.1f) [ms]\n", deviceTime, threadMin, threadMax);
			checkError(cudaThreadExit());
		}
	};
	
	vector<gpu> gpus;
	
	void ini(int type){
		// init the size of hit buffer
		d.hnum=0;
		pmax=0, pmxo=0, pn=0;
		
		for(vector<gpu>::iterator i=gpus.begin(); i!=gpus.end(); i++){
			i->set();
			i->ini(type); if(xgpu) sv++;
			d.hnum+=i->d.hnum;
			pmax+=i->pmax, pmxo+=i->pmxo;
		}
		
		{
			unsigned long size=d.hnum*sizeof(hit);
			checkError(cudaMallocHost((void**) &q.hits, size));
		}
		
		if(d.type==0){
			unsigned long size=pmxo*sizeof(photon);
			checkError(cudaMallocHost((void**) &q.pz, size));
		}
	}
	
	void fin(){
		for(vector<gpu>::iterator i=gpus.begin(); i!=gpus.end(); i++) i->set(), i->fin();
		checkError(cudaFreeHost(q.hits));
		if(d.type==0) checkError(cudaFreeHost(q.pz));
	}
	
	void listDevices(){
		int deviceCount, driver, runtime;
		cudaGetDeviceCount(&deviceCount);
		cudaDriverGetVersion(&driver);
		cudaRuntimeGetVersion(&runtime);
		fprintf(stderr, "Found %d devices, driver %d, runtime %d\n", deviceCount, driver, runtime);
		for(int device=0; device<deviceCount; ++device){
			cudaDeviceProp prop; cudaGetDeviceProperties(&prop, device);
			fprintf(stderr, "%d(%d.%d): %s %g GHz G(%lu) S(%lu) C(%lu) R(%d) W(%d)\n"
			"\tl%d o%d c%d h%d i%d m%d a%lu M(%lu) T(%d: %d,%d,%d) G(%d,%d,%d)\n",
			device, prop.major, prop.minor, prop.name, prop.clockRate/1.e6,
			(unsigned long)prop.totalGlobalMem, (unsigned long)prop.sharedMemPerBlock,
			(unsigned long)prop.totalConstMem, prop.regsPerBlock, prop.warpSize,
			prop.kernelExecTimeoutEnabled, prop.deviceOverlap, prop.computeMode,
			prop.canMapHostMemory, prop.integrated, prop.multiProcessorCount,
			(unsigned long)prop.textureAlignment,
			(unsigned long)prop.memPitch, prop.maxThreadsPerBlock,
			prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2],
			prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
		}
		fprintf(stderr, "\n");
	}
	
	static unsigned int old=0;
	#endif
	
	void print();
	void closeFile();
	void setNameWithGeometry(int x, int y, int z);
	
	void kernel(unsigned int num){
		#ifdef XCPU
		unsigned int & old = num;
		#endif
		if(old>0){
			d.hidx=0;
#ifdef XCPU
			for(d.blockIdx=0, d.gridDim=numBlock, blockDim.x=threadPerBlock; d.blockIdx<d.gridDim; d.blockIdx++)
			for(threadIdx.x=0; threadIdx.x<blockDim.x; threadIdx.x++) propagate(e, num);
			if(d.hidx>=d.hnum){ d.hidx=d.hnum; cerr<<"Error: data buffer overflow occurred!"<<endl; }
#else
			for(vector<gpu>::iterator i=gpus.begin(); i!=gpus.end(); i++) {
				i->set();
				i->kernel_i();
			}
#endif
#ifndef CAMERA
			cerr<<"photons: "<<old<<"  hits: "<<d.hidx<<endl;
#endif
		}
		
#ifndef XCPU
		{
			unsigned int over=d.type == 0 ? OVER : 1, sum=0;
			for(vector<gpu>::iterator i=gpus.begin(); i!=gpus.end(); i++){
				i->num=over*((num*(unsigned long long) i->pmax)/(over*(unsigned long long) pmax));
				sum+=i->num;
			}
			while(num>sum){
				static int res=0;
				gpu& g=gpus[res++%gpus.size()];
				if(g.num<g.pmax) g.num+=over, sum+=over;
			}
		}
		
		if(d.type==0){
			unsigned int idx=0;
			for(vector<gpu>::iterator i=gpus.begin(); i!=gpus.end(); i++) i->set(), i->kernel_c(idx);
		}
		
		for(vector<gpu>::iterator i=gpus.begin(); i!=gpus.end(); i++) i->set(), i->kernel_f();
#endif
		
		if(old>0) print();
#ifndef XCPU
		old=num;
		for(vector<gpu>::iterator i=gpus.begin(); i!=gpus.end(); i++) i->old=i->num;
#endif
	}
	
	float square(float x){
		return x*x;
	}
	
	/*
		init the flasher position and FWID 
	*/
	int flset(int str, int dom){
		int type=1;
		float r[3]={0, 0, 0};
		
		if(str<0){ type=2; str=-str; }
		if(str==0) switch(dom){
			case 1: type=3; r[0]=544.07; r[1]=55.89; r[2]=136.86; break;
			case 2: type=4; r[0]=11.87; r[1]=179.19; r[2]=-205.64; break;
		}
		else {
			for(int n=0; n<d.gsize; n++) {
				if(q.names[n].str==str && q.names[n].dom==dom){
				d.fla=n;
				for(int m=0; m<3; m++) r[m]=q.oms[n].r[m]; break;
				}
			}
		}
		
		for(int m=0; m<3; m++) 
			d.r[m]=r[m];
		
		float fwid=9.7f;
		{
			char * FWID=getenv("FWID");
			if(FWID!=NULL){ fwid=atof(FWID);
				cerr<<"Setting flasher beam width to "<<fwid<<" degrees"<<endl;
			}
		}
		
		if(fwid<0) d.ka=-1, d.up=0; else
		switch(type){
			case 1: d.ka=square(fcv*fwid); d.up=fcv*0.0f; break;
			case 2: d.ka=square(fcv*fwid); d.up=fcv*48.f; break;
			case 3: d.ka=0.0f;  d.up=fcv*(90.0f-41.13f);  break;
			case 4: d.ka=0.0f;  d.up=fcv*(41.13f-90.0f);  break;
		}
		
		return type;
	}
	
	void flini(int str, int dom){
		d.type = flset(str, dom);
		ini(d.type);
	}
	
	#ifdef XLIB
	const DOM& flget(int str, int dom){
		static DOM om;
		flset(str, dom); ini(0);
		for(int m=0; m<3; m++) om.r[m]=d.r[m];
		return om;
	}
	
	void flshift(float r[], float n[], float *m = NULL){
		float sft[3]={0};
		
		if(d.ka>0){
			float FLZ, FLR;
			sincosf(fcv*30.f, &FLZ, &FLR);
			FLZ*=OMR, FLR*=OMR;
			sft[0]+=FLR*n[0];
			sft[1]+=FLR*n[1];
			sft[2]+=FLZ;
			r[3]+=OMR*d.ocv;
		}
		
		float xi;
		sincosf(d.up, &n[2], &xi);
		n[0]*=xi; n[1]*=xi;
		
		if(m!=NULL){
			float o[3]={0,0,1};
			float r[3];
			
			r[0]=m[1]*o[2]-m[2]*o[1]; // m[1]
			r[1]=m[2]*o[0]-m[0]*o[2]; //-m[0]
			r[2]=m[0]*o[1]-m[1]*o[0]; // 0
			
			float norm=sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
			if(norm>0){
				float cs=0;
				for(int i=0; i<3; i++) r[i]/=norm, cs+=o[i]*m[i];
				float sn=sin(acos(cs)); //norm
				
				float R[3][3]={0};
				for(int i=0; i<3; i++)
				for(int j=0; j<3; j++)
				R[i][j]=(i==j?cs:sn*r[3-i-j]*((j-i+3)%3==1?1:-1))+(1-cs)*r[i]*r[j];
				
				float tmp[3];
				for(int i=0; i<3; i++){
					tmp[i]=0;
					for(int j=0; j<3; j++) tmp[i]+=R[i][j]*n[j];
				}
				for(int i=0; i<3; i++) n[i]=tmp[i];
				
				for(int i=0; i<3; i++){
					tmp[i]=0;
					for(int j=0; j<3; j++) tmp[i]+=R[i][j]*sft[j];
				}
				for(int i=0; i<3; i++) sft[i]=tmp[i];
				
			}
		}
		
		for(int i=0; i<3; i++) r[i]+=sft[i];
	}
	#endif
	
	void flone(unsigned long long num){
		for(long long i=llroundf(num*(long double)d.eff); i>0; i-=pmax) kernel(min(i, (long long) pmax));
		#ifndef XCPU
		kernel(0);
		#endif
	}
	
	void flasher(int str, int dom, unsigned long long num, int itr){
		flini(str, dom);
#ifdef CAMERA
		setNameWithGeometry(x_diff, y_diff, z_diff);
#endif
		for(int j=0; j<max(1, itr); j++){
			flone(num);
			if(itr>0) printf("\n");
		}
		
		fin();
#ifdef CAMERA
		closeFile();
#endif
	}

	void setOrder(int order){
		photonOrder = order;
	}
	
	#ifdef XCPU
	void start(){}
	void stop(){}
	void choose(int device){
		sv+=device;
		seed=device;
		numBlock=numBlock, threadPerBlock=threadPerBlock;
	}
	void listDevices(){}
	#else
	
	/*
	Before init the cuda, set the flag to configure sync
	*/
	void start(){
		cudaSetDeviceFlags(cudaDeviceBlockingSync);
	}
	
	void stop(){
		fprintf(stderr, "\n");
		for(vector<gpu>::iterator i=gpus.begin(); i!=gpus.end(); i++) i->set(), i->stop();
	}
	
	void choose(int device){
		if(device<0){
			int deviceCount; cudaGetDeviceCount(&deviceCount);
			for(int device=0; device<deviceCount; ++device){
				gpus.push_back(gpu(device));
				if(gpus.back().npho<=0) gpus.pop_back();
			}
		}
		else{
			sv+=device;
			gpus.push_back(gpu(device));
			if(gpus.back().npho<=0) gpus.pop_back();
		}
		if(gpus.size()<=0){
			cerr<<"No active GPU(s) selected!"<<endl;
			exit(5);
		}
		xgpu=gpus.size()>1;
	}
	#endif
	
	#include "f2k.cxx"
}

#ifndef XLIB
using namespace xppc;

int main(int arg_c, char *arg_a[]){
	start();
	if(arg_c<=1){
		listDevices();
		fprintf(stderr, "Use: %s [device] (f2k muons)\n"
		"     %s [str] [om] [num] [device] (flasher)\n", arg_a[0], arg_a[0]);
	}
	else if(0==strcmp(arg_a[1], "-")){
		initialize();
		ices & w = z.w[WNUM/2];
		cerr<<"For wavelength="<<w.wvl<<" [nm]  np="<<(1/w.coschr)<<"  cm="<<1/w.ocm<<" [m/ns]"<<endl;
		#ifdef TILT
		float4 r;
		r.w=0;
		if(arg_c==4){
			r.x=atof(arg_a[2]);
			r.y=atof(arg_a[3]);
		}
		else r.x=0, r.y=0;
		#endif
		for(int i=0; i<d.size; i++){
			float z=d.hmin+d.dh*i;
			#ifdef TILT
			r.z=z; for(int j=0; j<10; j++) r.z=z+zshift(d, r); z=r.z;
			#endif
			cout<<z<<" "<<w.z[i].abs<<" "<<w.z[i].sca*(1-d.g)<<endl;
		}
	}
	else if(arg_c<=2){
		int device=0;
		if(arg_c>1) device=atoi(arg_a[1]);
		initialize();
		choose(device);
		fprintf(stderr, "Processing f2k muons from stdin on device %d\n", device);
		f2k();
	}

	// Main operation for flasher
	else{
		int ledStr=0, ledDom=0, device=0, itr=0;
		unsigned long long num=1000000ULL;
		
		if(arg_c>1) ledStr=atoi(arg_a[1]);
		if(arg_c>2) ledDom=atoi(arg_a[2]);
		if(arg_c>3){
			num=(unsigned long long) atof(arg_a[3]);
			char * sub = strchr(arg_a[3], '*');
			if(sub!=NULL) itr=(int) atof(++sub);
		}
		if(arg_c>4) device=atoi(arg_a[4]);
		int order = log10(num);
		// cout << order << endl;
		initialize();
		// listDevices();
		setOrder(order);
		choose(device);
		fprintf(stderr, "Running flasher simulation on device %d\n", device);
		flasher(ledStr, ledDom, num, itr);
	}
	
	stop();

}
#endif