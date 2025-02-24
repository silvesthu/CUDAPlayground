#include "Common.h"

#include <cuda_runtime.h>
#include <optix.h>

struct Params
{
    float4*         image;
    unsigned int    image_width;
    unsigned int    image_height;
};

struct RayGenData
{
    float dummy;
};