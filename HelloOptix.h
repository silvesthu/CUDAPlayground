// Based on optixHello in SDK

#include "CommonOptix.h"

#include <cuda.h>                               // CUcontext
#include <math_constants.h>                     // CUDART_PI_F

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>

#include <iomanip>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>

// From optixSDKCurrent.cpp
#define OPTIX_CHECK( call )                                                                                            \
    {                                                                                                                  \
        OptixResult res = call;                                                                                        \
        if( res != OPTIX_SUCCESS )                                                                                     \
        {                                                                                                              \
            fprintf( stderr, "Optix call (%s) failed with code %d\n", #call, res );                                    \
            exit( 2 );                                                                                                 \
        }                                                                                                              \
    }

namespace HelloOptix
{
    static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */)
    {
        std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
    }

    template <typename T>
    struct SbtRecord
    {
        __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        T data;
    };

    typedef SbtRecord<RayGenData> RayGenSbtRecord;
    typedef SbtRecord<int>        MissSbtRecord;

    static constexpr unsigned int cWidth = 4;
    static constexpr unsigned int cHeight = 4;

	void Run()
	{
        printf("** HelloOptix **\n\n");

        //
        // Initialize CUDA and create OptiX context
        //
        OptixDeviceContext context = nullptr;
        {
            // Initialize CUDA
            CUDA_CHECK(cudaFree(0));

            CUcontext cuCtx = 0;  // zero means take the current context
            OPTIX_CHECK(optixInit());
            OptixDeviceContextOptions options = {};
            options.logCallbackFunction = &context_log_cb;
            options.logCallbackLevel = 4;
            OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
        }

        //
        // Create module
        //
        OptixModule module = nullptr;
        OptixPipelineCompileOptions pipeline_compile_options = {};
        {
            OptixModuleCompileOptions module_compile_options = {};
#if OPTIX_DEBUG_DEVICE_CODE
            module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
            module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif
            pipeline_compile_options.usesMotionBlur = false;
            pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
            pipeline_compile_options.numPayloadValues = 2;
            pipeline_compile_options.numAttributeValues = 2;
            pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;  // TODO: should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
            pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

            std::vector<char> binary;
            {
                std::ifstream file;
                file.open("HelloOptix.ptx", std::ios::binary);
                file.seekg(0, std::ios::end);
                size_t length = file.tellg();
                file.seekg(0, std::ios::beg);
                binary.resize(length);
                file.read(binary.data(), length);
                file.close();
            }

            OPTIX_CHECK(optixModuleCreate(
                context,
                &module_compile_options,
                &pipeline_compile_options,
                binary.data(),
                binary.size(),
                nullptr, nullptr,
                &module
            ));
        }

        //
        // Create program groups, including NULL miss and hitgroups
        //
        OptixProgramGroup raygen_prog_group = nullptr;
        OptixProgramGroup miss_prog_group = nullptr;
        {
            OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

            OptixProgramGroupDesc raygen_prog_group_desc = {}; //
            raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            raygen_prog_group_desc.raygen.module = module;
            raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__draw_solid_color";
            OPTIX_CHECK(optixProgramGroupCreate(
                context,
                &raygen_prog_group_desc,
                1,   // num program groups
                &program_group_options,
                nullptr, nullptr,
                &raygen_prog_group
            ));

            // Leave miss group's module and entryfunc name null
            OptixProgramGroupDesc miss_prog_group_desc = {};
            miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
            OPTIX_CHECK(optixProgramGroupCreate(
                context,
                &miss_prog_group_desc,
                1,   // num program groups
                &program_group_options,
                nullptr, nullptr,
                &miss_prog_group
            ));
        }

        //
        // Link pipeline
        //
        OptixPipeline pipeline = nullptr;
        {
            const uint32_t    max_trace_depth = 0;
            OptixProgramGroup program_groups[] = { raygen_prog_group };

            OptixPipelineLinkOptions pipeline_link_options = {};
            pipeline_link_options.maxTraceDepth = max_trace_depth;
            OPTIX_CHECK(optixPipelineCreate(
                context,
                &pipeline_compile_options,
                &pipeline_link_options,
                program_groups,
                sizeof(program_groups) / sizeof(program_groups[0]),
                nullptr, nullptr,
                &pipeline
            ));

            OptixStackSizes stack_sizes = {};
            for (auto& prog_group : program_groups)
            {
                OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes, pipeline));
            }

            uint32_t direct_callable_stack_size_from_traversal;
            uint32_t direct_callable_stack_size_from_state;
            uint32_t continuation_stack_size;
            OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
                0,  // maxCCDepth
                0,  // maxDCDEpth
                &direct_callable_stack_size_from_traversal,
                &direct_callable_stack_size_from_state, &continuation_stack_size));
            OPTIX_CHECK(optixPipelineSetStackSize(pipeline, direct_callable_stack_size_from_traversal,
                direct_callable_stack_size_from_state, continuation_stack_size,
                2  // maxTraversableDepth
            ));
        }

        //
        // Set up shader binding table
        //
        OptixShaderBindingTable sbt = {};
        {
            CUdeviceptr  raygen_record;
            const size_t raygen_record_size = sizeof(RayGenSbtRecord);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&raygen_record), raygen_record_size));
            RayGenSbtRecord rg_sbt;
            OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
            rg_sbt.data.dummy = CUDART_PI_F;
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>(raygen_record),
                &rg_sbt,
                raygen_record_size,
                cudaMemcpyHostToDevice
            ));

            CUdeviceptr miss_record;
            size_t      miss_record_size = sizeof(MissSbtRecord);
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&miss_record), miss_record_size));
            MissSbtRecord ms_sbt;
            OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>(miss_record),
                &ms_sbt,
                miss_record_size,
                cudaMemcpyHostToDevice
            ));

            sbt.raygenRecord = raygen_record;
            sbt.missRecordBase = miss_record;
            sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
            sbt.missRecordCount = 1;
        }

        //
        // launch
        //
        float4* output_device = nullptr;
        {
            CUstream stream;
            CUDA_CHECK(cudaStreamCreate(&stream));

            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&output_device), cWidth * cHeight * sizeof(float4)));
            Params params;
            params.image = output_device;
            params.image_width = cWidth;

            CUdeviceptr d_param;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_param), sizeof(Params)));
            CUDA_CHECK(cudaMemcpy(
                reinterpret_cast<void*>(d_param),
                &params, sizeof(params),
                cudaMemcpyHostToDevice
            ));

            OPTIX_CHECK(optixLaunch(pipeline, stream, d_param, sizeof(Params), &sbt, cWidth, cHeight, /*depth=*/1));
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_param)));
        }

        //
        // Output
        //
        {
            std::vector<float4> output_host;
            output_host.resize(cWidth * cHeight);

            CUDA_CHECK(cudaMemcpy(output_host.data(), output_device, output_host.size() * sizeof(float4), cudaMemcpyDeviceToHost));

            printf("Output = \n");
            for (int h = 0; h < cHeight; h++)
            {
                for (int w = 0; w < cWidth; w++)
                {
                    float4 data = output_host[h * cWidth + w];
                    printf("(%.2f %.2f %.2f %.2f), ", data.x, data.y, data.z, data.w);
                }
                printf("\n");
            }
        }

        //
        // Cleanup
        //
        {
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(output_device)));

            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.raygenRecord)));
            CUDA_CHECK(cudaFree(reinterpret_cast<void*>(sbt.missRecordBase)));

            OPTIX_CHECK(optixPipelineDestroy(pipeline));
            OPTIX_CHECK(optixProgramGroupDestroy(miss_prog_group));
            OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group));
            OPTIX_CHECK(optixModuleDestroy(module));

            OPTIX_CHECK(optixDeviceContextDestroy(context));
        }

        printf("\n");
	}
}
