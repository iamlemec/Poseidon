#ifndef _POSEIDON_KERNEL_h_
#define _POSEIDON_KERNEL_h_

#include <vector_types.h>

extern "C" void Poseidon_kernel(uchar4 *dst, uchar4 *dst_old, float4 *field, const int imageW, const int imageH, bool advance, bool pulse, const int px, const int py, const int fx, const int fy, float rfact, float tfact, float width, float steep, const int fieldType, bool draw_field, bool calc_field);

#endif
