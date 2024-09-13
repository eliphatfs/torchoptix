#define Py_LIMITED_API PY_VERSION_HEX
#include <stdio.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <Python.h>
#include "optixinc.h"
#include "optixdevptx.h"

PyObject* torchoptix_module_ref;
static OptixDeviceContext ocontext = nullptr;
static OptixModule omodule = nullptr;
static OptixProgramGroup oprograms[3] = { nullptr, nullptr, nullptr };
static OptixPipeline opipeline = nullptr;
static int ologlevel = 4;

PyDoc_STRVAR(torchoptix_log_level_doc, "set_log_level(level)\n\
\n\
Set OptiX log level (0-4).");

PyObject* torchoptix_log_level(PyObject* self, PyObject* args) {
    unsigned long long level;

    if (!PyArg_ParseTuple(args, "K", &level))
        return nullptr;

    if (level > 4 || level < 0)
    {
        PyErr_SetString(PyExc_ValueError, "TorchOptiX: invalid log level (0-4 allowed).");
        return nullptr;
    }

    ologlevel = level;
    Py_RETURN_NONE;
}

static void optix_log_callback(unsigned int level, const char* tag, const char* message, void* cbdata)
{
    if (level <= ologlevel)
    {
        printf("[OptiX] [%s: %d] %s\n", tag, level, message);
        fflush(stdout);
    }
}

inline bool check_cuda(CUresult result)
{
    if (result != CUDA_SUCCESS) {
        const char* errorStr;
        cuGetErrorString(result, &errorStr);
        PyErr_SetString(PyExc_RuntimeError, errorStr);
        return false;
    }
    return true;
}

inline bool ensure_initialize_context()
{
    if (!ocontext)
    {
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = &optix_log_callback;
        options.logCallbackLevel = ologlevel;
        if (optixInit() != OPTIX_SUCCESS)
        {
            PyErr_SetString(PyExc_RuntimeError, "OptiX initialize failed.");
            return false;
        }
        if (optixDeviceContextCreate(NULL, &options, &ocontext) != OPTIX_SUCCESS)
        {
            PyErr_SetString(PyExc_RuntimeError, "OptiX create context failed.");
            return false;
        }
    }
    OptixModuleCompileOptions moduleCompileOptions = {};
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    OptixPipelineLinkOptions pipelineLinkOptions = {};
    if (!omodule)
    {
        moduleCompileOptions.maxRegisterCount = 50;
        moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
        moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

        pipelineCompileOptions = {};
        pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipelineCompileOptions.usesMotionBlur = false;
        pipelineCompileOptions.numPayloadValues = 2;
        pipelineCompileOptions.numAttributeValues = 2;
        pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

        pipelineLinkOptions.maxTraceDepth = 2;

        char log[2048] = "OptiX create module failed: ";
        size_t sizeof_log = sizeof(log) - strlen(log);

        if (optixModuleCreateFromPTX(ocontext,
            &moduleCompileOptions,
            &pipelineCompileOptions,
            (const char*)generated_torchoptixdev_ptx,
            generated_torchoptixdev_ptx_len,
            log + strlen(log),      // Log string, concat after message
            &sizeof_log,            // Log string size
            &omodule
        ) != OPTIX_SUCCESS)
        {
            PyErr_SetString(PyExc_RuntimeError, log);
            return false;
        };
    }
    if (!oprograms[0])
    {
        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};
        pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        pgDesc.raygen.module = omodule;
        pgDesc.raygen.entryFunctionName = "__raygen__rg";

        // OptixProgramGroup raypg;
        char log[2048] = "OptiX create raygen program failed: ";
        size_t sizeof_log = sizeof(log) - strlen(log);
        if (optixProgramGroupCreate(ocontext,
            &pgDesc,
            1,
            &pgOptions,
            log, &sizeof_log,
            &oprograms[0]
        ) != OPTIX_SUCCESS)
        {
            PyErr_SetString(PyExc_RuntimeError, log);
            return false;
        };
    }
    if (!oprograms[1])
    {
        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};
        pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        pgDesc.raygen.module = omodule;
        pgDesc.raygen.entryFunctionName = "__miss__far";

        // OptixProgramGroup raypg;
        char log[2048] = "OptiX create miss program failed: ";
        size_t sizeof_log = sizeof(log) - strlen(log);
        if (optixProgramGroupCreate(ocontext,
            &pgDesc,
            1,
            &pgOptions,
            log, &sizeof_log,
            &oprograms[1]
        ) != OPTIX_SUCCESS)
        {
            PyErr_SetString(PyExc_RuntimeError, log);
            return false;
        };
    }
    if (!oprograms[2])
    {
        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc pgDesc = {};
        pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        pgDesc.hitgroup.moduleCH = omodule;
        pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";

        char log[2048] = "OptiX create hit program failed: ";
        size_t sizeof_log = sizeof(log) - strlen(log);
        if (optixProgramGroupCreate(ocontext,
            &pgDesc,
            1,
            &pgOptions,
            log, &sizeof_log,
            &oprograms[2]
        ) != OPTIX_SUCCESS)
        {
            PyErr_SetString(PyExc_RuntimeError, log);
            return false;
        };
    }
    if (!opipeline)
    {
        char log[2048] = "OptiX create pipeline failed: ";
        size_t sizeof_log = sizeof(log) - strlen(log);
        if (optixPipelineCreate(ocontext,
            &pipelineCompileOptions,
            &pipelineLinkOptions,
            oprograms,
            3,
            log, &sizeof_log,
            &opipeline
        ) != OPTIX_SUCCESS)
        {
            PyErr_SetString(PyExc_RuntimeError, log);
            return false;
        }

        if (optixPipelineSetStackSize(
            /* [in] The pipeline to configure the stack size for */
            opipeline,
            /* [in] The direct stack size requirement for direct
               callables invoked from IS or AH. */
            1024,
            /* [in] The direct stack size requirement for direct
               callables invoked from RG, MS, or CH.  */
            2 * 1024,
            /* [in] The continuation stack requirement. */
            2 * 1024,
            /* [in] The maximum depth of a traversable graph
               passed to trace. */
            1
        ) != OPTIX_SUCCESS)
        {
            PyErr_SetString(PyExc_RuntimeError, "OptiX pipeline set stack size failed.");
            return false;
        }
    }
    return true;
}

inline bool build_acceleration_structure(void* verts, void* tris, unsigned long long nverts, unsigned long long ntris, OptixTraversableHandle& out_handle, CUdeviceptr& out_pointer)
{
    // ==================================================================
    // triangle inputs
    // ==================================================================
    OptixBuildInput triangleInput = {};
    triangleInput.type
        = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    // create local variables, because we need a *pointer* to the
    // device pointers
    CUdeviceptr d_vertices = (CUdeviceptr)verts;
    CUdeviceptr d_indices = (CUdeviceptr)tris;

    triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleInput.triangleArray.vertexStrideInBytes = sizeof(float) * 3;
    triangleInput.triangleArray.numVertices = (unsigned int)nverts;
    triangleInput.triangleArray.vertexBuffers = &d_vertices;

    triangleInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangleInput.triangleArray.indexStrideInBytes = sizeof(int) * 3;
    triangleInput.triangleArray.numIndexTriplets = (unsigned int)ntris;
    triangleInput.triangleArray.indexBuffer = d_indices;

    uint32_t triangleInputFlags[1] = { 0 };

    // in this example we have one SBT entry, and no per-primitive
    // materials:
    triangleInput.triangleArray.flags = triangleInputFlags;
    triangleInput.triangleArray.numSbtRecords = 1;
    triangleInput.triangleArray.sbtIndexOffsetBuffer = 0;
    triangleInput.triangleArray.sbtIndexOffsetSizeInBytes = 0;
    triangleInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;

    // ==================================================================
    // BLAS setup
    // ==================================================================

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accelOptions.motionOptions.numKeys = 1;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes blasBufferSizes;
    if (optixAccelComputeMemoryUsage(ocontext,
        &accelOptions,
        &triangleInput,
        1,  // num_build_inputs
        &blasBufferSizes
    ) != OPTIX_SUCCESS)
    {
        PyErr_SetString(PyExc_RuntimeError, "OptiX acceleration structure compute memory usage failed.");
        return false;
    };

    // ==================================================================
    // execute build (main stage)
    // ==================================================================
    
    CUdeviceptr tempBuffer;
    if (!check_cuda(cuMemAlloc(&tempBuffer, blasBufferSizes.tempSizeInBytes))) return false;
    if (!check_cuda(cuMemAlloc(&out_pointer, blasBufferSizes.outputSizeInBytes))) return false;

    if(optixAccelBuild(ocontext,
        0,
        &accelOptions,
        &triangleInput,
        1,
        (CUdeviceptr)tempBuffer,
        blasBufferSizes.tempSizeInBytes,
        (CUdeviceptr)out_pointer,
        blasBufferSizes.outputSizeInBytes,
        &out_handle,
        nullptr, 0
    ) != OPTIX_SUCCESS)
    {
        PyErr_SetString(PyExc_RuntimeError, "OptiX acceleration structure build failed.");
        return false;
    };
    if (!check_cuda(cuCtxSynchronize())) return false;

    // ==================================================================
    // aaaaaand .... clean up
    // ==================================================================
    if (!check_cuda(cuMemFree(tempBuffer))) return false;
    return true;
}

PyDoc_STRVAR(torchoptix_build_doc, "build(verts, tris, n_verts, n_tris) -> handle\n\
\n\
Build OptiX acceleration structure.");

PyObject* torchoptix_build(PyObject* self, PyObject* args) {
    unsigned long long verts, tris, nverts, ntris;

    if (!PyArg_ParseTuple(args, "KKKK", &verts, &tris, &nverts, &ntris))
        return nullptr;

    if (!ensure_initialize_context())
        return nullptr;

    OptixTraversableHandle handle;
    CUdeviceptr pointer;
    if (!build_acceleration_structure((void*)verts, (void*)tris, nverts, ntris, handle, pointer))
        return nullptr;

    PyObject* res = PyTuple_New(2);
    PyTuple_SetItem(res, 0, PyLong_FromUnsignedLongLong((unsigned long long)handle));
    PyTuple_SetItem(res, 1, PyLong_FromUnsignedLongLong((unsigned long long)pointer));
    return res;
}

PyDoc_STRVAR(torchoptix_release_doc, "release(handle)\n\
\n\
Release OptiX acceleration structure.");

PyObject* torchoptix_release(PyObject* self, PyObject* args) {
    PyObject* handle;

    if (!PyArg_ParseTuple(args, "O", &handle))
        return nullptr;

    if (!ensure_initialize_context())
        return nullptr;

    if (!check_cuda(cuMemFree((CUdeviceptr)PyLong_AsUnsignedLongLong(PyTuple_GetItem(handle, 1)))))
        return nullptr;
    Py_RETURN_NONE;
}

inline bool trace_rays(OptixTraversableHandle handle, void* rays_o, void* rays_d, void* out_t, void* out_i, float t_max, unsigned long long n_rays)
{
    OptixShaderBindingTable sbt = {};

    constexpr int sbt_record_size = (OPTIX_SBT_RECORD_HEADER_SIZE / OPTIX_SBT_RECORD_ALIGNMENT + 1) * OPTIX_SBT_RECORD_ALIGNMENT;
    char record[sbt_record_size * 3];
    optixSbtRecordPackHeader(oprograms[0], record);
    optixSbtRecordPackHeader(oprograms[1], record + sbt_record_size);
    optixSbtRecordPackHeader(oprograms[2], record + sbt_record_size * 2);

    CUdeviceptr dsbt;
    if (!check_cuda(cuMemAlloc(&dsbt, sbt_record_size * 3))) return false;
    if (!check_cuda(cuMemcpyHtoD(dsbt, record, sbt_record_size * 3))) return false;

    sbt.raygenRecord = dsbt;
    sbt.missRecordBase = dsbt + sbt_record_size;
    sbt.hitgroupRecordBase = dsbt + sbt_record_size * 2;
    sbt.missRecordStrideInBytes = sbt.hitgroupRecordStrideInBytes = sbt_record_size;
    sbt.missRecordCount = sbt.hitgroupRecordCount = 1;

    LaunchParams p = {};
    p.rays_o = (unsigned long long)rays_o;
    p.rays_d = (unsigned long long)rays_d;
    p.out_t = (unsigned long long)out_t;
    p.out_i = (unsigned long long)out_i;
    p.t_max = t_max;
    p.traversable = handle;

    CUdeviceptr dp;
    if (!check_cuda(cuMemAlloc(&dp, sizeof(LaunchParams)))) return false;
    if (!check_cuda(cuMemcpyHtoD(dp, &p, sizeof(LaunchParams)))) return false;

    if (optixLaunch(/*! pipeline we're launching launch: */
        opipeline, 0,
        /*! parameters and SBT */
        (CUdeviceptr)dp,
        sizeof(LaunchParams),
        &sbt,
        /*! dimensions of the launch: */
        n_rays,
        1,
        1
    ) != OPTIX_SUCCESS)
    {
        PyErr_SetString(PyExc_RuntimeError, "OptiX launch failed.");
        return false;
    }

    if (!check_cuda(cuCtxSynchronize())) return false;
    if (!check_cuda(cuMemFree(dp))) return false;
    if (!check_cuda(cuMemFree(dsbt))) return false;

    return true;
}

PyDoc_STRVAR(torchoptix_trace_doc, "trace_rays(handle, rays_o, rays_d, out_t, out_i, t_max, n_rays)\n\
\n\
Trace rays with OptiX.");

PyObject* torchoptix_trace(PyObject* self, PyObject* args) {
    /* Shared references that do not need Py_DECREF before returning. */
    PyObject* handle;
    unsigned long long rays_o, rays_d, out_t, out_i, n_rays;
    float t_max;

    /* Parse positional and keyword arguments */
    if (!PyArg_ParseTuple(args, "OKKKKfK", &handle, &rays_o, &rays_d, &out_t, &out_i, &t_max, &n_rays))
        return nullptr;

    if (!ensure_initialize_context())
        return nullptr;

    OptixTraversableHandle ohandle = (OptixTraversableHandle)PyLong_AsUnsignedLongLong(PyTuple_GetItem(handle, 0));

    if (!trace_rays(ohandle, (void*)rays_o, (void*)rays_d, (void*)out_t, (void*)out_i, t_max, n_rays))
        return nullptr;

    Py_RETURN_NONE;
}

/*
 * List of functions to add to torchoptix in exec_torchoptix().
 */
static PyMethodDef torchoptix_functions[] = {
    { "build", (PyCFunction)torchoptix_build, METH_VARARGS, torchoptix_build_doc },
    { "release", (PyCFunction)torchoptix_release, METH_VARARGS, torchoptix_release_doc },
    { "set_log_level", (PyCFunction)torchoptix_log_level, METH_VARARGS, torchoptix_log_level_doc },
    { "trace_rays", (PyCFunction)torchoptix_trace, METH_VARARGS, torchoptix_trace_doc },
    { NULL, NULL, 0, NULL }  /* marks end of array */
};

/*
 * Initialize torchoptix. May be called multiple times, so avoid
 * using static state.
 */
int exec_torchoptix(PyObject *module) {
    torchoptix_module_ref = module;
    PyModule_AddFunctions(module, torchoptix_functions);

    PyModule_AddStringConstant(module, "__author__", "eliphatfs");
    PyModule_AddStringConstant(module, "__version__", "0.0.1");
    PyModule_AddIntConstant(module, "year", 2024);

    return 0;  /* success */
}

/*
 * Documentation for torchoptix.
 */
PyDoc_STRVAR(torchoptix_doc, "Modular OptiX ray tracing functions interop with PyTorch.");


static PyModuleDef_Slot torchoptix_slots[] = {
    { Py_mod_exec, (void*)exec_torchoptix },
    { 0, NULL }
};

static PyModuleDef torchoptix_def = {
    PyModuleDef_HEAD_INIT,
    "torchoptix",
    torchoptix_doc,
    0,              /* m_size */
    NULL,           /* m_methods */
    torchoptix_slots,
    NULL,           /* m_traverse */
    NULL,           /* m_clear */
    NULL,           /* m_free */
};

extern "C" {
    PyMODINIT_FUNC PyInit_torchoptix() {
        return PyModuleDef_Init(&torchoptix_def);
    }
}
