/*this function is used to generate the backward projection*/
//includes,system
#include <iostream>
#include <math.h>
using namespace std;

//includes,cuda
#include "cuda.h"
#include "cuda_runtime.h"

//includes,matlab
#include "mex.h"

#define HalfFanShiftDirection (1)
#define PI (3.141592653F)

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
void BackwardProjection(float *Object, int ObjectElements_X, int ObjectElements_Y, int ObjectElements_Z, float VSize_X, float VSize_Y, float VSize_Z,
	                                        float *Projection, int ProjectionElements_X, int ProjectionElements_Y, int ProjectionElements_Z, 
 										    float *ViewAngles, int BlkSizeX, int BlkSizeY, int BlkSizeZ, float IsoToSource, 
											int TotalSubset,int IndexSubset, int SizeSubset,float max_gama,float min_gama,float gama);

__global__ void backwardProj(float *Object, int ObjectElements_X, int ObjectElements_Y, int ObjectElements_Z, float VSize_X, float VSize_Y, float VSize_Z,
	                                                 int ProjectionElements_X, int ProjectionElements_Y, int ProjectionElements_Z,                                            
	                                                 float *ViewAngles, float IsoToSource, 
													 int TotalSubset,int IndexSubset, int SizeSubset,float max_gama,float min_gama ,float gama);

// texture parameters
texture<float, 3, cudaReadModeElementType> tex_backward;

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

	//projection
	int IMAGER_X = mxGetM(prhs[0]); 
	int NumberofProjeciton_h = mxGetN(prhs[0]); 
	input = mxGetPr(prhs[0]);
	
	if( !mxIsDouble(prhs[0]) || mxIsComplex(prhs[0]) || (NumberofProjeciton_h == 1 && IMAGER_X == 1) )
		mexErrMsgTxt("projection data must be double and noncomplex matrix !"); 

	//the projection angle
	int angleM = mxGetM(prhs[1]);
	int angleN = mxGetN(prhs[1]);
	int nViews = angleM * angleN;
	D_angleMat = mxGetPr(prhs[1]);

	if(!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1]) || (angleM != 1 && angleN != 1))
		mexErrMsgTxt("please choose the correct projection angle !");

	if(NumberofProjeciton_h != nViews)
		mexErrMsgTxt("please confirm whether projection data and projection angle match !");

	//image and detector size
	int paraM = mxGetM(prhs[2]);
	int paraN = mxGetN(prhs[2]);
	int paraNum = paraM * paraN;
	para = mxGetPr(prhs[2]);
	
	int VOLUME_X = (int)para[0];
	int VOLUME_Y = (int)para[1]; 
	float FOV = (float)para[2];
	float DistanceIsoSource = (float)para[3];
	float max_gama = (float)para[4];
	float min_gama = (float)para[5];
	
	float VOXELSIZE_X = FOV/(float)VOLUME_X;
	float VOXELSIZE_Y = FOV/(float)VOLUME_Y;
	float gama = (max_gama - min_gama)/IMAGER_X;
	
	if(!mxIsDouble(prhs[2]) || mxIsComplex(prhs[2]) || paraNum != 6)
		mexErrMsgTxt("please enter the 6 parameters about image size !");

	if(VOLUME_X <= 0 || VOLUME_Y <= 0 || VOXELSIZE_X <= 0 || VOXELSIZE_Y <= 0 || DistanceIsoSource <= 0 )
		mexErrMsgTxt("geometric parameters must be positive !");

	//output
	mxbuffer = mxCreateDoubleMatrix(VOLUME_X,VOLUME_Y, mxREAL); 
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
	int SizeSubset = NumberofProjeciton_h;

	//loading projection and copy projection from host to device
	float *h_projection = (float*) mxMalloc (sizeof(float) * IMAGER_X * NumberofProjeciton_h );
	for(int i = 0;i < IMAGER_X * NumberofProjeciton_h;i++ )
	{
		h_projection[i] = (float) input[i];
	}

	float *Projection;
	gpuErrchk( cudaMalloc((void **) &Projection, sizeof(float) * IMAGER_X * IMAGER_Y * NumberofProjeciton_h) );
	gpuErrchk( cudaMemcpy(Projection, h_projection, sizeof(float) * IMAGER_X * IMAGER_Y * NumberofProjeciton_h, cudaMemcpyHostToDevice) );
	mxFree(h_projection);

	// loading view angle infomation and copy angle information from host to device
	float *angle = (float *)mxMalloc(sizeof(float) * NumberofProjeciton_h);
	for(int i = 0;i < NumberofProjeciton_h;i++)
	{
		angle[i] = (float)D_angleMat[i];
	}

	float *ViewAngles;
	gpuErrchk( cudaMalloc((void **) &ViewAngles, sizeof(float) * NumberofProjeciton_h) );
	gpuErrchk( cudaMemcpy(ViewAngles, angle, sizeof(float) * NumberofProjeciton_h, cudaMemcpyHostToDevice) );
	mxFree(angle);

	//image information
	float *Volume;
	gpuErrchk( cudaMalloc((void **) &Volume, sizeof(float) * VOLUME_X * VOLUME_Y * VOLUME_Z) );
	gpuErrchk( cudaMemset(Volume, 0, sizeof(float) * VOLUME_X * VOLUME_Y * VOLUME_Z) );
	
	// performing forward projection
	BackwardProjection(Volume, VOLUME_X, VOLUME_Y, VOLUME_Z, VOXELSIZE_X, VOXELSIZE_Y, VOXELSIZE_Z,
		                                Projection, IMAGER_X, IMAGER_Y, NumberofProjeciton_h, 
		                               ViewAngles, BlockSizeX, BlockSizeY, BlockSizeZ, DistanceIsoSource,
									   TotalSubset,IndexSubset,SizeSubset,max_gama,min_gama,gama);

	// output data
	float *image = (float *)mxMalloc(sizeof(float) * VOLUME_X * VOLUME_Y * VOLUME_Z);
	gpuErrchk( cudaMemcpy(image, Volume, sizeof(float) * VOLUME_X * VOLUME_Y * VOLUME_Z, cudaMemcpyDeviceToHost) );
	
	for(int i = 0;i < VOLUME_X*VOLUME_Y;i++)
	{
		output[i] = (double) image[i];
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

	mxFree(image);
	gpuErrchk( cudaFree(ViewAngles));
	gpuErrchk( cudaFree(Projection));
	gpuErrchk( cudaFree(Volume));
}

void BackwardProjection(float *Object, int ObjectElements_X, int ObjectElements_Y, int ObjectElements_Z, float VSize_X, float VSize_Y, float VSize_Z,
	                                        float *Projection, int ProjectionElements_X, int ProjectionElements_Y, int ProjectionElements_Z,
 										    float *ViewAngles, int BlkSizeX, int BlkSizeY, int BlkSizeZ, float IsoToSource, 
											int TotalSubset,int IndexSubset,int SizeSubset,float max_gama,float min_gama,float gama)
{
	//cuda parameters
	dim3 BlockSize;
	BlockSize.x = BlkSizeX;
	BlockSize.y = BlkSizeY;
	BlockSize.z = BlkSizeZ;

	dim3 GridSize;
	GridSize.x = (ObjectElements_X - 1) / BlkSizeX + 1;
	GridSize.y = (ObjectElements_Y - 1) / BlkSizeY + 1;
	GridSize.z = (ObjectElements_Z - 1) / BlkSizeZ + 1;

	// 3D cuda Array
	cudaArray *CUDA3DArray = 0;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaMemcpy3DParms copyParams = {0};
	cudaExtent extent;

	// create cuda 3D array
	extent = make_cudaExtent(ProjectionElements_X, ProjectionElements_Y, SizeSubset);
	gpuErrchk( cudaMalloc3DArray(&CUDA3DArray, &channelDesc, extent) );

	// copy parameters
	copyParams.extent = extent;
	copyParams.kind = cudaMemcpyDeviceToDevice;
	copyParams.srcPtr = make_cudaPitchedPtr((void *) Projection, extent.width * sizeof(float), extent.width, extent.height);
	copyParams.dstArray = CUDA3DArray;
	gpuErrchk( cudaMemcpy3D(&copyParams) );

	tex_backward.filterMode = cudaFilterModePoint;

	// perform backward projection
	cudaBindTextureToArray(tex_backward, CUDA3DArray, channelDesc);
	backwardProj<<<GridSize, BlockSize>>>(Object, ObjectElements_X, ObjectElements_Y, ObjectElements_Z, VSize_X, VSize_Y, VSize_Z, 
		                                                                    ProjectionElements_X, ProjectionElements_Y, ProjectionElements_Z,                                                   
		                                                                    ViewAngles, IsoToSource,
																			TotalSubset, IndexSubset,SizeSubset,max_gama,min_gama,gama);
	gpuErrchk( cudaUnbindTexture(tex_backward) );
	gpuErrchk( cudaFreeArray(CUDA3DArray) );
	gpuErrchk( cudaGetLastError() );
}


__global__ void backwardProj(float *Object, int ObjectElements_X, int ObjectElements_Y, int ObjectElements_Z, float VSize_X, float VSize_Y, float VSize_Z,
	                                                 int ProjectionElements_X, int ProjectionElements_Y, int ProjectionElements_Z,                                 
	                                                 float *ViewAngles, float IsoToSource, 
													 int TotalSubset,int IndexSubset,int SizeSubset,float max_gama,float min_gama,float gama)
{
	// obtain current id on thread
	const int tx = blockIdx.x * blockDim.x + threadIdx.x;
	const int ty = blockIdx.y * blockDim.y + threadIdx.y;
	const int tz = blockIdx.z * blockDim.z + threadIdx.z; 

	if ((tx < ObjectElements_X) && (ty < ObjectElements_Y) && (tz < ObjectElements_Z) )
	{
		float result = 0.0F;

		for(int iproj = IndexSubset; iproj < ProjectionElements_Z; iproj += TotalSubset)
		{
			// setup rotation angle
			float sinTheta = sin( ViewAngles[iproj] );
			float cosTheta = cos( ViewAngles[iproj] );

			// coordinate of a point in the phantom in rectangular coordinate
			float xptemp = (tx - ObjectElements_X/2 + 0.5F) * VSize_X;
			float yptemp = (ty - ObjectElements_Y/2 + 0.5F) * VSize_Y;

			// coordinate of a point in the phantom in rotated coordinate
			float xp = xptemp * cosTheta + yptemp * sinTheta;
			float yp = - xptemp * sinTheta + yptemp * cosTheta;

			// coordinate of source in rotated coordinate
			float xs = -IsoToSource;
			float ys = 0.0F;
			
			//float x = DetectorDirection * (ys + (yp - ys)*(IsoToDectector - xs) / (xp - xs) + HalfFanShiftDirection * HalfFanShift);
			float x = (max_gama + atan((yp - ys)/(xp - xs))) /gama - 0.5F;
			
			// coordinate on the imager in unit of pixsize
			int xi = floor(x);

			//float factor1 = sqrt((xp-xs)*(xp-xs)+(yp-ys)*(yp-ys)) / abs(xp-xs);
			//float factor2 = (IsoToSource+IsoToDectector) * (IsoToSource+IsoToDectector) / (xp - xs) / (xp-xs);
			float factor = 1/(xp-xs);
			
			// obtain values at four nearest neighbors
			if (x>=0 && x<=ProjectionElements_X)
			{
				int OrderSubsetProjection = (iproj - IndexSubset) / TotalSubset;

				float v00 = tex3D(tex_backward, xi+(xi<0), 0, OrderSubsetProjection);
				float v10 = tex3D(tex_backward, xi+1-(xi+1>=ProjectionElements_X), 0, OrderSubsetProjection);		

				x -= xi;

				float value = v00*(1-x) + v10*x;
				result += factor * factor * value;
				
				//result += factor1*factor2*value;
			}
			
		}
		Object[tz * ObjectElements_X * ObjectElements_Y + ty * ObjectElements_X+ tx] = result ;//* VSize_X * VSize_Y / DSize_X;
	}

	return;
}
