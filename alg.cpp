#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <iosfwd>
#include <string>
#include <random>
#include <ctime>
#include <chrono>

#ifdef __APPLE__
    #include "OpenCL/opencl.h"
#else
    #include "CL/cl.h"
#endif

using namespace std;
using namespace std::chrono;


float randMToN(float M, float N)
{
    return M + (rand() / ( RAND_MAX / (N-M) ) ) ;
}

const char *kernelSource =                                      "\n" \
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable                    \n" \
"__kernel void matrix_multiplication(  __global float *a,                       \n" \
"                       __global float *b,                       \n" \
"                       __global float *c,                       \n" \
"                       const unsigned int width_a,                   \n" \
"                       const unsigned int width_b)                    \n" \
"{                                                      \n" \
"    int y = get_global_id(0);                                  \n" \
"    int x = get_global_id(1);                                  \n" \
"    float sum = 0; \n"\
"    for(int index = 0; index < width_a; index++){ \n"\
"       sum += a[y * width_a + index] * b[index * width_b + x]; \n"\
"    }  \n"\
"    //printf(\"%d %f \\n\", width_b * y + x, sum); \n"\
"    c[width_b * y + x] = sum;                \n" \
"}                                                               \n" \
                                                                "\n" ;

void executeKernel(bool use_gpu, float* matrix_a, float* matrix_b, unsigned int size, double *results){
  // Device input buffers
  cl_mem d_a;
  cl_mem d_b;
  // Device output buffer
  cl_mem d_c;

  cl_platform_id cpPlatform;        // OpenCL platform
  cl_device_id device_id;           // device ID
  cl_context context;               // context
  cl_command_queue queue;           // command queue
  cl_program program;               // program
  cl_kernel kernel;                 // kernel

  cl_int error;
  cl_build_status status;
  FILE* programHandle;
  char *programBuffer; char *programLog;
  size_t programSize; size_t logSize;

  // Initialize matrices on host
  float* result_matrix = new float[size * size];

  size_t bytes_matrix_a = size * size * sizeof(float);
  size_t bytes_matrix_b = size * size * sizeof(float);
  size_t bytes_result_matrix = size * size * sizeof(float);

  high_resolution_clock::time_point begin = high_resolution_clock::now();

  // Bind to platform
  clGetPlatformIDs(1, &cpPlatform, NULL);

  // Get ID for the device
  cl_device_type device_type = use_gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;
  clGetDeviceIDs(cpPlatform, device_type, 2, &device_id, NULL);

  // Create a context
  context = clCreateContext(0, 1, &device_id, NULL, NULL, NULL);

  // Create a command queue
  queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, NULL);

  // Create the compute program from the source buffer
  program = clCreateProgramWithSource(context, 1, (const char **) & kernelSource, NULL, NULL);

  // Build the program executable
  error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

  if (error != CL_SUCCESS) {
      // check build error and build status first
      clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_STATUS,
              sizeof(cl_build_status), &status, NULL);

      // check build log
      clGetProgramBuildInfo(program, device_id,
              CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);
      programLog = (char*) calloc (logSize+1, sizeof(char));
      clGetProgramBuildInfo(program, device_id,
              CL_PROGRAM_BUILD_LOG, logSize+1, programLog, NULL);
      printf("Build failed; error=%d, status=%d, programLog:nn%s",
              error, status, programLog);
      free(programLog);
  }


  // Create the compute kernel in the program we wish to run
  kernel = clCreateKernel(program, "matrix_multiplication", NULL);

  // Create the input and output arrays in device memory for our calculation
  d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes_matrix_a, NULL, NULL);
  d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes_matrix_b, NULL, NULL);
  d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes_result_matrix, NULL, NULL);

  cl_event buffer_a_event, buffer_b_event, buffer_c_event;


  // Write our data set into the input array in device memory
  clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, bytes_matrix_a, matrix_a, 0, NULL, &buffer_a_event);
  clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0, bytes_matrix_b, matrix_b, 0, NULL, &buffer_b_event);

  // Set the arguments to our compute kernel
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
  clSetKernelArg(kernel, 3, sizeof(unsigned int), &size);
  clSetKernelArg(kernel, 4, sizeof(unsigned int), &size);

  size_t y_range = size;
  size_t x_range = size;
  size_t global[2] = {y_range, x_range};
  size_t local[2] = {NULL, NULL};

  // Execute the kernel over the entire range of the data set
  clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, NULL);

  // Wait for the command queue to get serviced before reading back results
  clFinish(queue);

  // Read the results from the device
  clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, bytes_result_matrix, result_matrix, 0, NULL, &buffer_c_event);

  high_resolution_clock::time_point end = high_resolution_clock::now();
  // release OpenCL resources
  clReleaseMemObject(d_a);
  clReleaseMemObject(d_b);
  clReleaseMemObject(d_c);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  long duration = duration_cast<microseconds>( end - begin ).count() * 1000;

  //release host memory
  free(result_matrix);

  cl_ulong time_start, time_end;
  double buffer_a_time, buffer_b_time, buffer_c_time;

  clGetEventProfilingInfo(buffer_a_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
  clGetEventProfilingInfo(buffer_a_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
  buffer_a_time = (time_end - time_start) / 1000000.0;

  clGetEventProfilingInfo(buffer_b_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
  clGetEventProfilingInfo(buffer_b_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
  buffer_b_time = (time_end - time_start) / 1000000.0;

  clGetEventProfilingInfo(buffer_c_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
  clGetEventProfilingInfo(buffer_c_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);
  buffer_c_time = (time_end - time_start) / 1000000.0;

  results[0] = (double)duration / CLOCKS_PER_SEC;
  results[1] = buffer_a_time;
  results[2] = buffer_b_time;
  results[3] = buffer_c_time;
}
