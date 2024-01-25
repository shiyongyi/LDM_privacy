//includes,system
#include <iostream>
#include <math.h>
using namespace std;

//includes,cuda
#include "cuda.h"
#include "cuda_runtime.h"

//includes,matlab
#include "mex.h"

#define PI (3.141592653)
#define TOL (1e-4)
#define HalfFanShiftDirection (1)

//function for check error
#define gpuErrchk(ans) {gpuAssert((ans), __FILE__, __LINE__);}

inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
    {
      fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
    }
}

//function declaration
void ForwardProjection(float *Projection, int ProjectionElements_X, int ProjectionElements_Y, int ProjectionElements_Z,
	                                      float *Object, int ObjectElements_X, int ObjectElements_Y, int ObjectElements_Z, float VSize_X, float VSize_Y, float VSize_Z,
										  float *ViewAngles, int BlkSizeX, int BlkSizeY, int BlkSizeZ, float IsoToSource,
										  const int TotalSubset, const int IndexSubset,float max_gama,float min_gama,float gama);

__global__ void forwardProj(float *dest, int ProjectionElements_X, int ProjectionElements_Y, int ProjectionElements_Z, 
                                                   int ObjectElements_X, int ObjectElements_Y, int ObjectElements_Z, float VSize_X, float VSize_Y, float VSize_Z,
												   float *ViewAngles, float IsoToSource, 
												   const int TotalSubset, const int IndexSubset,float max_gama,float min_gama,float gama);


__device__ float forwardProjRay(float alphaMin, float alphaMax, int startDim,
	                                                     int direction[2], float dAlpha[2], float length,
	                                                     float xi, float yi, float xs, float ys, 
														 int ObjectElements_X, int ObjectElements_Y, int ObjectElements_Z, float VSize_X, float VSize_Y, float VSize_Z);
// texture parameters
texture<float, 3, cudaReadModeElementType> tex_forward;

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] ) 
{
	double *input,*output;
	double *D_angleMat;
	double *para;
	double *parabuffer;
	
	mxArray *mxbuffer;

	if(nrhs != 3) 
		mexErrMsgTxt("the number of input parameters must be 3 !"); 

	if(nlhs != 1)
		mexErrMsgTxt("the number of output parameters must be 1 !");

	//image
	int VOLUME_X = mxGetM(prhs[0]); 
	int VOLUME_Y = mxGetN(prhs[0]); 
	input = mxGetPr(prhs[0]);
	
	if( !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) || (VOLUME_X == 1 && VOLUME_Y == 1) )
		mexErrMsgTxt("image data must be double and noncomplex matrix !"); 

	//the projection angle
	int angleM = mxGetM(prhs[1]);
	int angleN = mxGetN(prhs[1]);
	int NumberofProjeciton_h = angleM * angleN;
	D_angleMat = mxGetPr(prhs[1]);

	if(!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) || (angleM != 1 && angleN != 1))
		mexErrMsgTxt("please choose the correct projection angle !");

	//image and detector size
	int paraM = mxGetM(prhs[2]);
	int paraN = mxGetN(prhs[2]);
	int paraNum = paraM * paraN;
	para = mxGetPr(prhs[2]);

	float IMAGER_X = (int)para[0];
	float FOV = (float)para[1];
	float DistanceIsoSource = (float)para[2];
	float max_gama = (float)para[3];
	float min_gama = (float)para[4];
	
	float VOXELSIZE_X = FOV/(float)VOLUME_X;
	float VOXELSIZE_Y = FOV/(float)VOLUME_Y;
	float gama = (max_gama - min_gama)/IMAGER_X ;
	
	if(!mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]) || paraNum != 5)
		mexErrMsgTxt("please enter the 5 parameters about image size !");
		
	//check error
	if(VOXELSIZE_X <= 0 || VOXELSIZE_Y <= 0 || IMAGER_X <= 0 ||  DistanceIsoSource <= 0  )
		mexErrMsgTxt("geometric parameters must be positive !");
   
	//output
	mxbuffer = mxCreateDoubleMatrix(IMAGER_X,NumberofProjeciton_h, mxREAL); 
	output = mxGetPr(mxbuffer); 

	//cuda parameters
	int BlockSizeX = 16;
	int BlockSizeY = 1;
	int BlockSizeZ = 16;
	
	//the sanme parameters in 2D reconstruction
	int VOLUME_Z = 1;
	int IMAGER_Y = 1;
	float VOXELSIZE_Z = 1;

	int TotalSubset = 1;
	int IndexSubset = 0;

	//load volume information
	float *buffer = (float *)mxMalloc(sizeof(float) * VOLUME_X * VOLUME_Y * VOLUME_Z);
	for(int i = 0;i < VOLUME_X * VOLUME_Y * VOLUME_Z;i++ )
	{
		buffer[i] = (float) input[i];
	}

	// copy to device-end
	float *Volume;
	gpuErrchk( cudaMalloc((void **) &Volume, sizeof(float) * VOLUME_X * VOLUME_Y * VOLUME_Z) );
	gpuErrchk( cudaMemcpy(Volume, buffer, sizeof(float) * VOLUME_X * VOLUME_Y * VOLUME_Z, cudaMemcpyHostToDevice) );
	mxFree(buffer);

	// copy to device-end for view angle information
	float *angle = (float*)mxMalloc(sizeof(float) * NumberofProjeciton_h);
	for(int i = 0;i < NumberofProjeciton_h;i++)
	{
		angle[i] = (float)D_angleMat[i];
	}

	float *ViewAngles;
	gpuErrchk( cudaMalloc((void **) &ViewAngles, sizeof(float) * NumberofProjeciton_h) );
	gpuErrchk( cudaMemcpy(ViewAngles, angle, sizeof(float) * NumberofProjeciton_h, cudaMemcpyHostToDevice) );
	mxFree(angle);

	// performing forward projection
	float *Projection;
	gpuErrchk( cudaMalloc((void **) &Projection, sizeof(float) * IMAGER_X * IMAGER_Y * NumberofProjeciton_h) );
	gpuErrchk( cudaMemset(Projection, 0, sizeof(float) * IMAGER_X * IMAGER_Y * NumberofProjeciton_h) );

	ForwardProjection(Projection, IMAGER_X, IMAGER_Y, NumberofProjeciton_h, 
		Volume, VOLUME_X, VOLUME_Y, VOLUME_Z, VOXELSIZE_X, VOXELSIZE_Y, VOXELSIZE_Z,
		ViewAngles, BlockSizeX, BlockSizeY, BlockSizeZ, DistanceIsoSource,
		TotalSubset,IndexSubset,max_gama,min_gama,gama);

	// outputData
	float *h_projection = (float *)mxMalloc(sizeof(float) * IMAGER_X * IMAGER_Y * NumberofProjeciton_h);
	gpuErrchk( cudaMemcpy(h_projection, Projection, sizeof(float) * IMAGER_X * IMAGER_Y * NumberofProjeciton_h, cudaMemcpyDeviceToHost) );
	
	for(int i = 0;i < IMAGER_X * IMAGER_Y * NumberofProjeciton_h;i++)
	{
		output[i] = (double)h_projection[i];
	}

	// matlab struct
	const char *FieldNames[] = {"version", "copyright", "data", "projection", "volumes", "fov", "detectors", "distance", "angle"};
	const char *VersionValue[] = {"version: 2.1"};
	const char *CopyrightValue[] = {"2015-2017 the Institute of Image processing and Pattern recognition, Xi'an Jiaotong University"};
		
	plhs[0] = mxCreateStructMatrix(1,1,9,FieldNames);
	
	mxSetField(plhs[0],0,"data",mxbuffer);
	
	mxbuffer = mxCreateString(VersionValue[0]);
	mxSetField(plhs[0], 0, "version", mxbuffer);
	
	mxbuffer = mxCreateString(CopyrightValue[0]);
	mxSetField(plhs[0], 0, "copyright", mxbuffer);
	
	mxbuffer = mxCreateDoubleMatrix(1,NumberofProjeciton_h, mxREAL); 
	parabuffer = mxGetPr(mxbuffer); 
	for(int i = 0; i < NumberofProjeciton_h; i++)
	{
		parabuffer[i] = D_angleMat[i];
	}
	mxSetField(plhs[0], 0, "projection", mxbuffer);
	
	mxbuffer = mxCreateDoubleMatrix(1,2,mxREAL);
	parabuffer = mxGetPr(mxbuffer);
	parabuffer[0] = VOLUME_X;
	parabuffer[1] = VOLUME_Y;
	mxSetField(plhs[0], 0, "volumes",mxbuffer);
	
	mxbuffer = mxCreateDoubleMatrix(1,1,mxREAL);
	parabuffer = mxGetPr(mxbuffer);
	parabuffer[0] = FOV;
	mxSetField(plhs[0], 0, "fov",mxbuffer);
	
	mxbuffer = mxCreateDoubleMatrix(1,1,mxREAL);
	parabuffer = mxGetPr(mxbuffer);
	parabuffer[0] = IMAGER_X;
	mxSetField(plhs[0], 0, "detectors", mxbuffer);
	
	mxbuffer = mxCreateDoubleMatrix(1,1,mxREAL);
	parabuffer = mxGetPr(mxbuffer);
	parabuffer[0] = DistanceIsoSource;
	mxSetField(plhs[0], 0, "distance",mxbuffer);
	
	mxbuffer = mxCreateDoubleMatrix(1,2,mxREAL);
	parabuffer = mxGetPr(mxbuffer);
	parabuffer[0] = min_gama;
	parabuffer[1] = max_gama;
	mxSetField(plhs[0], 0, "angle", mxbuffer);
	
	mxFree(h_projection);
	gpuErrchk( cudaFree(Volume));
	gpuErrchk( cudaFree(ViewAngles));
	gpuErrchk( cudaFree(Projection) );
}


void ForwardProjection(float *Projection, int ProjectionElements_X, int ProjectionElements_Y, int ProjectionElements_Z, 
	                                      float *Object, int ObjectElements_X, int ObjectElements_Y, int ObjectElements_Z, float VSize_X, float VSize_Y, float VSize_Z,
										  float *ViewAngles, int BlkSizeX, int BlkSizeY, int BlkSizeZ, float IsoToSource, 
										  int TotalSubset, int IndexSubset,float max_gama,float min_gama,float gama)
{

	dim3 BlockSize;
	BlockSize.x = BlkSizeX;
	BlockSize.y = BlkSizeY;
	BlockSize.z = BlkSizeZ;

	dim3 GridSize;
	GridSize.x = (ProjectionElements_X - 1) / BlkSizeX + 1;
	GridSize.y = (ProjectionElements_Y - 1) / BlkSizeY + 1;
	GridSize.z = (ProjectionElements_Z - 1) / BlkSizeZ + 1;

	// 3D cuda Array
	cudaArray *CUDA3DArray = 0;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaMemcpy3DParms copyParams = {0};
	cudaExtent extent;

	// creat cuda 3D array
	extent = make_cudaExtent(ObjectElements_X, ObjectElements_Y, ObjectElements_Z);
	gpuErrchk( cudaMalloc3DArray(&CUDA3DArray, &channelDesc, extent) );

	// copy parameters
	copyParams.extent = extent;
	copyParams.kind = cudaMemcpyDeviceToDevice;
	copyParams.srcPtr = make_cudaPitchedPtr((void *) Object, extent.width * sizeof(float), extent.width, extent.height);
	copyParams.dstArray = CUDA3DArray;
	gpuErrchk( cudaMemcpy3D(&copyParams) );

	tex_forward.filterMode = cudaFilterModePoint;

	cudaBindTextureToArray(tex_forward, CUDA3DArray, channelDesc);
	forwardProj<<<GridSize, BlockSize>>>(Projection, ProjectionElements_X, ProjectionElements_Y, ProjectionElements_Z, 
           		                                                         ObjectElements_X, ObjectElements_Y, ObjectElements_Z, VSize_X, VSize_Y, VSize_Z, 
		                                                                 ViewAngles, IsoToSource, 
																		 TotalSubset, IndexSubset,max_gama,min_gama,gama);
	gpuErrchk( cudaUnbindTexture(tex_forward) );
	gpuErrchk( cudaFreeArray(CUDA3DArray) );
	gpuErrchk( cudaGetLastError() );
}

__global__ void forwardProj(float *dest, int ProjectionElements_X, int ProjectionElements_Y, int ProjectionElements_Z, 
                                                   int ObjectElements_X, int ObjectElements_Y, int ObjectElements_Z, float VSize_X, float VSize_Y, float VSize_Z,
												   float *ViewAngles, float IsoToSource, 
												   const int TotalSubset, const int IndexSubset,float max_gama,float min_gama,float gama)
{
	// obtain current id on thread
	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;
	const int tz = blockIdx.z * blockDim.z + threadIdx.z; 

	if ( (tx < ProjectionElements_X) & (ty < ProjectionElements_Y) & (tz < ProjectionElements_Z) )
	{
		int iproj = IndexSubset + tz * TotalSubset;

		// setup rotation angle
		float sinTheta = sin( ViewAngles[iproj] );
		float cosTheta = cos( ViewAngles[iproj] );

		float xtemp = IsoToSource;
		float ytemp = tan(-max_gama + gama * (tx+ 0.5F) ) * 2 * IsoToSource;
		//float ytemp = DetectorDirection * (tx - ProjectionElements_X/2 + 0.5F) * DSize_X - HalfFanShiftDirection * HalfFanShift;

		// set coordinate for pixel on the imager
		float xi = xtemp * cosTheta - ytemp * sinTheta;
		float yi = xtemp * sinTheta + ytemp * cosTheta;

		// set coordinate for source point
		float xs = -IsoToSource * cosTheta;
		float ys = -IsoToSource * sinTheta;

		float temp;
		int startDim;
		float alphaMin = 0.0F;
		float alphaMax = 1.0F;
		float alpha1, alphan;

		// determin the alpha range, and start dimension
		alpha1 = (-ObjectElements_X/2 * VSize_X - xs) / (xi - xs);
		alphan = ( ObjectElements_X/2 * VSize_X - xs) / (xi - xs);
		temp = min(alpha1, alphan);
		alphaMin = max(alphaMin, temp);
		if(alphaMin == temp) startDim = 0;
		temp = max(alpha1, alphan);
		alphaMax = min(alphaMax, temp);

		alpha1 = (-ObjectElements_Y/2 * VSize_Y - ys)/(yi - ys);
		alphan = ( ObjectElements_Y/2 * VSize_Y - ys)/(yi - ys);
		temp = min(alpha1, alphan);
		alphaMin = max(alphaMin, temp);
		if(alphaMin == temp) startDim = 1;
		temp = max(alpha1, alphan);
		alphaMax = min(alphaMax, temp);

		// the total length from source to pixel
		float length = sqrt( (xs-xi)*(xs-xi)+(ys-yi)*(ys-yi) );

		// direction of the propagation
		int direction[2] = { (xi<xs)*(-2) + 1, (yi<ys)*(-2) + 1 };

		// determin the increment of the alphas
    	float dAlpha[2] = { VSize_X / abs(xi - xs), VSize_Y / abs(yi - ys) };

		// get forward projection along the ray line
		dest[tz * ProjectionElements_X * ProjectionElements_Y + ty * ProjectionElements_X + tx] = forwardProjRay(alphaMin,alphaMax,startDim,direction,dAlpha,length,
			xi,yi,xs,ys,ObjectElements_X, ObjectElements_Y, ObjectElements_Z, VSize_X, VSize_Y, VSize_Z);
	}
}

// forward projection along the rayline
__device__ float forwardProjRay(float alphaMin, float alphaMax, int startDim,
	                                                     int direction[3], float dAlpha[3], float length,
	                                                     float xi, float yi, float xs, float ys, 
	                                                     int ObjectElements_X, int ObjectElements_Y, int ObjectElements_Z, float VSize_X, float VSize_Y, float VSize_Z)
{
	float alpha = alphaMin;
	float alphax, alphay;
	int ix,iy,iz;
	iz = 0;

	if(startDim == 0)
	{
		if(direction[0] == 1)
			ix = 0;
		else
			ix = ObjectElements_X - 1;
		alphax = alpha + dAlpha[0];

		iy = (ys + alpha*(yi - ys)) / VSize_Y + ObjectElements_Y/2;
		alphay = ((iy + (direction[1] > 0) - ObjectElements_Y/2) * VSize_Y - ys) / (yi - ys);
	}
	else if(startDim == 1)
	{
		if(direction[1] == 1)
			iy = 0;
		else
			iy = ObjectElements_Y - 1;
		alphay = alpha + dAlpha[1];

		ix = (xs + alpha*(xi - xs)) / VSize_X + ObjectElements_X/2;
		alphax = ((ix + (direction[0] > 0) - ObjectElements_X/2) * VSize_X - xs) / (xi - xs);
	}

	// tracing the line while accumulate the projection
	float result = 0.0F;
	while(alpha < alphaMax - TOL)
	{
		if(alphax <= alphay)
		{
			result += length * (alphax-alpha) * tex3D(tex_forward, ix, iy, iz);

			alpha = alphax;
			alphax += dAlpha[0];
			ix += direction[0];
		}
		else
		{
			result += length * (alphay-alpha) * tex3D(tex_forward, ix, iy, iz);

			alpha = alphay;
			alphay += dAlpha[1];
			iy += direction[1];
		}
	}


	return result;
}
