#include <stdio.h>
#include "Poseidon_kernel.h"

#define CHAR_MASK 0x000000ff

inline int topBlock(int nel, int bs)
{
  return (nel+(bs-nel%bs))/bs;
}

__global__ void mainRun(uchar4 *dst, uchar4 *dst_old, float4* field, const int imageW, const int imageH, float rfact, float tfact)
{
  int ix = blockDim.x*blockIdx.x + threadIdx.x;
  int iy = blockDim.y*blockIdx.y + threadIdx.y;

  float cfact = 1.0;

  if ((ix < imageW) && (iy < imageH)) {
    int pixel = imageW * iy + ix;

    int4 cval;
    cval.x = dst_old[pixel].x;
    cval.y = dst_old[pixel].y;
    cval.z = dst_old[pixel].z;

    int4 val;
    val.x = 0;
    val.y = 0;
    val.z = 0;

    if (ix < imageW-1) {
      val.x += dst_old[pixel+1].x;
      val.y += dst_old[pixel+1].y;
      val.z += dst_old[pixel+1].z;
    } else {
      val.x += dst_old[pixel+1-imageW].x;
      val.y += dst_old[pixel+1-imageW].y;
      val.z += dst_old[pixel+1-imageW].z;
    }
    if (ix > 0) {
      val.x += dst_old[pixel-1].x;
      val.y += dst_old[pixel-1].y;
      val.z += dst_old[pixel-1].z;
    } else {
      val.x += dst_old[pixel-1+imageW].x;
      val.y += dst_old[pixel-1+imageW].y;
      val.z += dst_old[pixel-1+imageW].z;
    }
    if (iy < imageH-1) {
      val.x += dst_old[pixel+imageW].x;
      val.y += dst_old[pixel+imageW].y;
      val.z += dst_old[pixel+imageW].z;
    } else {
      val.x += dst_old[ix].x;
      val.y += dst_old[ix].y;
      val.z += dst_old[ix].z;
    }
    if (iy > 0) {
      val.x += dst_old[pixel-imageW].x;
      val.y += dst_old[pixel-imageW].y;
      val.z += dst_old[pixel-imageW].z;
    } else {
      val.x += dst_old[pixel+imageW*(imageH-1)].x;
      val.y += dst_old[pixel+imageW*(imageH-1)].y;
      val.z += dst_old[pixel+imageW*(imageH-1)].z;
    }

    float4 fval = field[pixel];

    val.x = (cfact*powf(cval.x,1.5)+val.x+rfact*fval.x)*tfact;
    val.y = (cfact*powf(cval.y,1.5)+val.y+rfact*fval.y)*tfact;
    val.z = (cfact*powf(cval.z,1.5)+val.z+rfact*fval.z)*tfact;

    dst[pixel].x = val.x & CHAR_MASK;
    dst[pixel].y = val.y & CHAR_MASK;
    dst[pixel].z = val.z & CHAR_MASK;
  }
}

__global__ void calcField(float4 *field, const int fx, const int fy, const int imageW, const int imageH, const int fieldType)
{
  int ix = blockDim.x*blockIdx.x + threadIdx.x;
  int iy = blockDim.y*blockIdx.y + threadIdx.y;

  if ((ix < imageW) && (iy < imageH)) {
    int pixel = imageW * iy + ix;

    float fval;
    if (fieldType == 0) {
      fval = sqrtf(powf(abs(ix-fx),2)+powf(abs(iy-fy),2));
    } else {
      fval = 15000.0*rsqrtf(powf(abs(ix-fx),2)+powf(abs(iy-fy),2)+1);
    }

    field[pixel].x = fval;
    field[pixel].y = fval;
    field[pixel].z = fval;
  }
}

__global__ void doPulse(uchar4 *dst, float4 *field, const int imageW, const int imageH, const int px, const int py)
{
  const float pfact = 100.0;

  int ix = blockDim.x*blockIdx.x + threadIdx.x;
  int iy = blockDim.y*blockIdx.y + threadIdx.y;

  if ((ix < imageW) && (iy < imageH)) {
    int pixel = imageW * iy + ix;

    float rad = pfact*rsqrtf(powf(abs(ix-px),2)+powf(abs(iy-py),2)+1);

    //dst[pixel].x += rad;
    //dst[pixel].y += rad;
    //dst[pixel].z += rad;

    field[pixel].x += rad;
    field[pixel].y += rad;
    field[pixel].z += rad;
  }
}

void Poseidon_kernel(uchar4 *dst, uchar4 *dst_old, float4 *field, const int imageW, const int imageH, bool advance, bool pulse, const int px, const int py, const int fx, const int fy, bool init, float rfact, float tfact, const int fieldType)
{
  dim3 threadsPerBlock(16,16);
  dim3 numBlocks(topBlock(imageW,threadsPerBlock.x),topBlock(imageH,threadsPerBlock.y));

  if (pulse) {
    if (advance) {
      doPulse<<<numBlocks, threadsPerBlock>>>(dst_old, field, imageW, imageH, px, py);
    } else {
      doPulse<<<numBlocks, threadsPerBlock>>>(dst, field, imageW, imageH, px, py);
    }
  }

  if (advance) {
    if (init) {
      calcField<<<numBlocks, threadsPerBlock>>>(field, fx, fy, imageW, imageH, fieldType);
    }
    mainRun<<<numBlocks, threadsPerBlock>>>(dst, dst_old, field, imageW, imageH, rfact, tfact);
  }

  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess)
  {
    printf("Poseidon_kernel error: %s\n", cudaGetErrorString(code));
  }
}
