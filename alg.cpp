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


double randMToN(double M, double N)
{
    return M + (rand() / ( RAND_MAX / (N-M) ) ) ;
}

const char *kernelSource =                                      "\n" \
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable                    \n" \
"__kernel void matrix_multiplication(  __global double *a,                       \n" \
"                       __global double *b,                       \n" \
"                       __global double *c,                       \n" \
"                       const unsigned int width_a,                   \n" \
"                       const unsigned int width_b)                    \n" \
"{                                                      \n" \
"    int y = get_global_id(0);                                  \n" \
"    int x = get_global_id(1);                                  \n" \
"    double sum = 0; \n"\
"    for(int index = 0; index < width_a; index++){ \n"\
"       sum += a[y * width_a + index] * b[index * width_b + x]; \n"\
"    }  \n"\
"    //printf(\"%d %f \\n\", width_b * y + x, sum); \n"\
"    c[width_b * y + x] = sum;                \n" \
"}                                                               \n" \
                                                                "\n" ;

int executeKernel(bool use_gpu, unsigned int width_matrix_a, unsigned int height_matrix_a, unsigned int width_matrix_b){
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
  unsigned int height_matrix_b = width_matrix_a;

  double* matrix_a = new double[width_matrix_a * height_matrix_a];
  double* matrix_b = new double[width_matrix_b * height_matrix_b];
  double* result_matrix = new double[height_matrix_a * width_matrix_b];

  int i;
  int j;

  for(i = 0; i < height_matrix_a; i++)
  {
    for(j = 0; j < width_matrix_a; j++)
    {
      matrix_a[i * width_matrix_a + j] = randMToN(1, 3);
    }
  }

  for(i = 0; i < height_matrix_b; i++)
  {
    for(j = 0; j < width_matrix_b; j++)
    {
      matrix_b[i * width_matrix_b + j] = randMToN(1, 3);
    }
  }

  size_t bytes_matrix_a = height_matrix_a * width_matrix_a * sizeof(double);
  size_t bytes_matrix_b = height_matrix_b * width_matrix_b * sizeof(double);
  size_t bytes_result_matrix = height_matrix_a * width_matrix_b * sizeof(double);

  high_resolution_clock::time_point begin = high_resolution_clock::now();

  // Bind to platform
  clGetPlatformIDs(1, &cpPlatform, NULL);

  // Get ID for the device
  cl_device_type device_type = use_gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;
  clGetDeviceIDs(cpPlatform, device_type, 2, &device_id, NULL);

  // Create a context
  context = clCreateContext(0, 1, &device_id, NULL, NULL, NULL);

  // Create a command queue
  queue = clCreateCommandQueue(context, device_id, 0, NULL);

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

  // Write our data set into the input array in device memory
  clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, bytes_matrix_a, matrix_a, 0, NULL, NULL);
  clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0, bytes_matrix_b, matrix_b, 0, NULL, NULL);

  // Set the arguments to our compute kernel
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
  clSetKernelArg(kernel, 3, sizeof(unsigned int), &width_matrix_a);
  clSetKernelArg(kernel, 4, sizeof(unsigned int), &width_matrix_b);

  size_t local_size = 1;
  size_t y_range = height_matrix_a * local_size;
  size_t x_range = width_matrix_b * local_size;
  size_t global[2] = {y_range, x_range};
  size_t local[2] = {1, 1};

  // Execute the kernel over the entire range of the data set
  clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, local, 0, NULL, NULL);

  // Wait for the command queue to get serviced before reading back results
  clFinish(queue);

  // Read the results from the device
  clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0, bytes_result_matrix, result_matrix, 0, NULL, NULL );

  high_resolution_clock::time_point end = high_resolution_clock::now();

  long duration = duration_cast<microseconds>( end - begin ).count();

  // for(i = 0; i < height_matrix_a * width_matrix_b; i++)
  // {
  //   cout << result_matrix[i] << "\t";
  //   if((i + 1) % width_matrix_b == 0)
  //     cout << endl;
  // }

  cout << result_matrix[height_matrix_a * width_matrix_b - 1] << endl;
  cout << "Elapsed Time: " << (double)duration / CLOCKS_PER_SEC  << endl;

  // release OpenCL resources
  clReleaseMemObject(d_a);
  clReleaseMemObject(d_b);
  clReleaseMemObject(d_c);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  //release host memory
  free(matrix_a);
  free(matrix_b);
  free(result_matrix);

  return 0;
}


int main( int argc, char** argv)
{
  //if(argc < 5){
  //  throw std::invalid_argument("Matrix Multiplication requires CPU/GPU choice and 3 size arguments.");
  //}

  bool use_gpu = atoi(argv[1]) == 1;
  unsigned int width_matrix_a = atoi(argv[2]);
  unsigned int height_matrix_a = atoi(argv[3]);
  unsigned int width_matrix_b = atoi(argv[4]);

  cout << width_matrix_a << endl;
  cout << height_matrix_a << endl;
  cout << width_matrix_b << endl;

  executeKernel(use_gpu, width_matrix_a, height_matrix_a, width_matrix_b);
}
