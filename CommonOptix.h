#include "Common.h"

#include <cuda_runtime.h>
#include <optix.h>

struct Params
{
    float4* image;
    unsigned int image_width;
};

struct RayGenData
{
    float dummy;
};