#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include "Poseidon_kernel.h"

#define CHAR_MASK 0x000000ff

inline int topBlock(int nel, int bs)
{
    return (nel+(bs-nel%bs))/bs;
}

__device__ float sigmoid(float x)
{
    return 0.5*(tanh(x)+1.0);
}

__device__ int4 neighbors(uchar4* dst_old, int pixel, int ix, int iy, int imageW, int imageH)
{
    // combinable offsets
    int off_left = -1 + (ix == 0)*imageW;
    int off_right = 1 - (ix == imageW - 1)*imageW;
    int off_up = -imageW + (iy == 0)*imageW*imageH;
    int off_down = imageW - (iy == imageH - 1)*imageW*imageH;

    int4 nval;
    nval.x = 0.0;
    nval.y = 0.0;
    nval.z = 0.0;

    // up left
    nval.x += dst_old[pixel+off_up+off_left].x;
    nval.y += dst_old[pixel+off_up+off_left].y;
    nval.z += dst_old[pixel+off_up+off_left].z;

    // up
    nval.x += dst_old[pixel+off_up].x;
    nval.y += dst_old[pixel+off_up].y;
    nval.z += dst_old[pixel+off_up].z;

    // up right
    nval.x += dst_old[pixel+off_up+off_right].x;
    nval.y += dst_old[pixel+off_up+off_right].y;
    nval.z += dst_old[pixel+off_up+off_right].z;

    // right
    nval.x += dst_old[pixel+off_right].x;
    nval.y += dst_old[pixel+off_right].y;
    nval.z += dst_old[pixel+off_right].z;

    // down right
    nval.x += dst_old[pixel+off_down+off_right].x;
    nval.y += dst_old[pixel+off_down+off_right].y;
    nval.z += dst_old[pixel+off_down+off_right].z;

    // down
    nval.x += dst_old[pixel+off_down].x;
    nval.y += dst_old[pixel+off_down].y;
    nval.z += dst_old[pixel+off_down].z;

    // down left
    nval.x += dst_old[pixel+off_down+off_left].x;
    nval.y += dst_old[pixel+off_down+off_left].y;
    nval.z += dst_old[pixel+off_down+off_left].z;

    // left
    nval.x += dst_old[pixel+off_left].x;
    nval.y += dst_old[pixel+off_left].y;
    nval.z += dst_old[pixel+off_left].z;

    return nval;
}

__global__ void drawUpdate(uchar4 *dst, uchar4 *dst_old, float4* field, int imageW, int imageH,
                           float rfact, float tfact, float width, float steep)
{
    int ix = blockDim.x*blockIdx.x + threadIdx.x;
    int iy = blockDim.y*blockIdx.y + threadIdx.y;

    if ((ix < imageW) && (iy < imageH)) {
        int pixel = imageW * iy + ix;

        // core value
        uchar4 cval = dst_old[pixel];

        // neighbor count
        int4 nval = neighbors(dst_old, pixel, ix, iy, imageW, imageH);

        // background field
        float4 fn = field[pixel];

        /* fizzlife */
        // float thresh = 1.5201;
        // float steep = 2.0;
        // val.x = CHAR_MASK*erff(steep*(val.x/CHAR_MASK-thresh));
        // val.y = CHAR_MASK*erff(steep*(val.y/CHAR_MASK-thresh));
        // val.z = CHAR_MASK*erff(steep*(val.z/CHAR_MASK-thresh));

        /* superfizz */
        // float cfact = 1.0;
        // val.x = (cfact*powf(cval.x,1.5)+val.x+rfact*fval.x)*tfact;
        // val.y = (cfact*powf(cval.y,1.5)+val.y+rfact*fval.y)*tfact;
        // val.z = (cfact*powf(cval.z,1.5)+val.z+rfact*fval.z)*tfact;

        /* conway's rule */
        /*
        int4 cn;
        cn.x = int(roundf(cval.x/CHAR_MASK));
        cn.y = int(roundf(cval.y/CHAR_MASK));
        cn.z = int(roundf(cval.z/CHAR_MASK));

        int4 nn;
        nn.x = int(roundf(val.x/CHAR_MASK));
        nn.y = int(roundf(val.y/CHAR_MASK));
        nn.z = int(roundf(val.z/CHAR_MASK));

        val.x = CHAR_MASK * (((cn.x == 1) && ((nn.x == 2) || (nn.x == 3))) ||
                             ((cn.x == 0) &&                 (nn.x == 3)));
        val.y = CHAR_MASK * (((cn.y == 1) && ((nn.y == 2) || (nn.y == 3))) ||
                             ((cn.y == 0) &&                 (nn.y == 3)));
        val.z = CHAR_MASK * (((cn.z == 1) && ((nn.z == 2) || (nn.z == 3))) ||
                             ((cn.z == 0) &&                 (nn.z == 3)));
        */

        /* quantum conway */
        float4 cn;
        cn.x = float(cval.x)/CHAR_MASK;
        cn.y = float(cval.y)/CHAR_MASK;
        cn.z = float(cval.z)/CHAR_MASK;

        float4 nn;
        nn.x = float(nval.x)/CHAR_MASK;
        nn.y = float(nval.y)/CHAR_MASK;
        nn.z = float(nval.z)/CHAR_MASK;

        float4 vn;
        vn.x = ( sigmoid(steep*(cn.x-0.5)) * (1.0 - sigmoid(steep*((2.0-width)-nn.x)) - sigmoid(steep*(nn.x-(3.0+width)))) )
             + ( sigmoid(steep*(0.5-cn.x)) * (1.0 - sigmoid(steep*((3.0-width)-nn.x)) - sigmoid(steep*(nn.x-(3.0+width)))) );
        vn.y = ( sigmoid(steep*(cn.y-0.5)) * (1.0 - sigmoid(steep*((2.0-width)-nn.y)) - sigmoid(steep*(nn.y-(3.0+width)))) )
             + ( sigmoid(steep*(0.5-cn.y)) * (1.0 - sigmoid(steep*((3.0-width)-nn.y)) - sigmoid(steep*(nn.y-(3.0+width)))) );
        vn.z = ( sigmoid(steep*(cn.z-0.5)) * (1.0 - sigmoid(steep*((2.0-width)-nn.z)) - sigmoid(steep*(nn.z-(3.0+width)))) )
             + ( sigmoid(steep*(0.5-cn.z)) * (1.0 - sigmoid(steep*((3.0-width)-nn.z)) - sigmoid(steep*(nn.z-(3.0+width)))) );

        /* field driver */
        float4 dn;
        // dn.x = vn.x;
        // dn.y = vn.y;
        // dn.z = vn.z;

        dn.x = (vn.x + rfact * fn.x) * tfact;
        dn.y = (vn.y + rfact * fn.y) * tfact;
        dn.z = (vn.z + rfact * fn.z) * tfact;

        // final output
        dst[pixel].x = int(CHAR_MASK * dn.x) & CHAR_MASK;
        dst[pixel].y = int(CHAR_MASK * dn.y) & CHAR_MASK;
        dst[pixel].z = int(CHAR_MASK * dn.z) & CHAR_MASK;
    }
}

__global__ void drawField(uchar4 *dst, float4* field, int imageW, int imageH)
{
    int ix = blockDim.x*blockIdx.x + threadIdx.x;
    int iy = blockDim.y*blockIdx.y + threadIdx.y;

    if ((ix < imageW) && (iy < imageH)) {
        int pixel = imageW * iy + ix;

        float4 fval = field[pixel];
        dst[pixel].x = int(CHAR_MASK * fval.x) & CHAR_MASK;
        dst[pixel].y = int(CHAR_MASK * fval.y) & CHAR_MASK;
        dst[pixel].z = int(CHAR_MASK * fval.z) & CHAR_MASK;
    }
}

__global__ void calcField(float4 *field, uchar4 *dst_old, int fx, int fy,
                          int imageW, int imageH, int fieldType)
{
    int ix = blockDim.x*blockIdx.x + threadIdx.x;
    int iy = blockDim.y*blockIdx.y + threadIdx.y;

    if ((ix < imageW) && (iy < imageH)) {
        int pixel = imageW * iy + ix;

        float4 fval;
        float rad = 50.0;
        float cur = 1.5;
        if (fieldType == 0) {
            fval.x = 0.0;
            fval.y = 0.0;
            fval.z = 0.0;
        } else if (fieldType == 1) {
            float fval0 = max(0.0,1.0-powf(powf(abs(float(ix-fx)/rad),cur)+powf(abs(float(iy-fy)/rad),cur),1.0/cur));
            fval.x = fval0;
            fval.y = fval0;
            fval.z = fval0;
        } else {
            uchar4 cval = dst_old[pixel];
            int4 nval = neighbors(dst_old, pixel, ix, iy, imageW, imageH);

            float4 cn;
            cn.x = float(cval.x)/CHAR_MASK;
            cn.y = float(cval.y)/CHAR_MASK;
            cn.z = float(cval.z)/CHAR_MASK;

            float4 nn;
            nn.x = float(nval.x)/CHAR_MASK;
            nn.y = float(nval.y)/CHAR_MASK;
            nn.z = float(nval.z)/CHAR_MASK;

            fval.x = max(0.0,nn.x-cn.x/8);
            fval.y = max(0.0,nn.y-cn.y/8);
            fval.z = max(0.0,nn.z-cn.z/8);

            // fval.x = 1.0 - sigmoid(2.0-nn.x) - sigmoid(nn.x-3.0);
            // fval.y = 1.0 - sigmoid(2.0-nn.x) - sigmoid(nn.x-3.0);
            // fval.z = 1.0 - sigmoid(2.0-nn.x) - sigmoid(nn.x-3.0);
        }

        field[pixel].x = fval.z;
        field[pixel].y = fval.x;
        field[pixel].z = fval.y;
    }
}

__global__ void doPulse(uchar4 *dst, float4 *field, int imageW, int imageH, int px, int py)
{
  int ix = blockDim.x*blockIdx.x + threadIdx.x;
  int iy = blockDim.y*blockIdx.y + threadIdx.y;

  if ((ix < imageW) && (iy < imageH)) {
    int pixel = imageW * iy + ix;

    float rad = 25.0;
    float cur = 1.5;
    float dist = max(0.0,1.0-powf(powf(abs(float(ix-px)/rad),cur)+powf(abs(float(iy-py)/rad),cur),1.0/cur));

    dst[pixel].x = (dst[pixel].x + int(CHAR_MASK*dist)) & CHAR_MASK;
    dst[pixel].y = (dst[pixel].y + int(CHAR_MASK*dist)) & CHAR_MASK;
    dst[pixel].z = (dst[pixel].z + int(CHAR_MASK*dist)) & CHAR_MASK;
  }
}

void Poseidon_kernel(uchar4 *dst, uchar4 *dst_old, float4 *field,
    int imageW, int imageH, bool advance, bool pulse, int px, int py, int fx, int fy,
    float rfact, float tfact, float width, float steep, int fieldType, bool draw_field, bool calc_field)
{
    dim3 threadsPerBlock(16,16);
    dim3 numBlocks(topBlock(imageW,threadsPerBlock.x),topBlock(imageH,threadsPerBlock.y));

    // printf("%f %f %f %f\n",rfact,tfact,width,steep);

    if (pulse) {
        if (advance) {
            doPulse<<<numBlocks, threadsPerBlock>>>(dst_old, field, imageW, imageH, px, py);
        } else {
            doPulse<<<numBlocks, threadsPerBlock>>>(dst, field, imageW, imageH, px, py);
        }
    }

    if (calc_field || (fieldType == 2)) {
        calcField<<<numBlocks, threadsPerBlock>>>(field, dst_old, fx, fy, imageW, imageH, fieldType);
    }

    if (draw_field) {
        drawField<<<numBlocks, threadsPerBlock>>>(dst, field, imageW, imageH);
    }

    if (advance) {
        drawUpdate<<<numBlocks, threadsPerBlock>>>(dst, dst_old, field,
            imageW, imageH, rfact, tfact, width, steep);
    }

    cudaError_t code = cudaGetLastError();
    if (code != cudaSuccess) {
        printf("Poseidon_kernel error: %s\n", cudaGetErrorString(code));
    }
}
