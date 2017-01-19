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
#include <vector>

#include "alg.cpp"
#include "matrix_generator.cpp"

using namespace std;


double calcMean(vector<float> results){
  double sum = accumulate(results.begin(), results.end(), 0.0);
  double mean = sum / results.size();

  return mean;
}

double calcStdev(vector<float> results){
  double mean = calcMean(results);
  double sq_sum = inner_product(results.begin(), results.end(), results.begin(), 0.0);
  double stdev = sqrt(sq_sum / results.size() - mean * mean);
  return stdev;
}

int main( int argc, char** argv)
{
  cout << "Arguments: MinSize, MaxSize, StepSize, Iterations, GPU Bit" << endl << std::flush;

  unsigned int min_size = atoi(argv[1]);
  unsigned int max_size = atoi(argv[2]);
  unsigned int step_size = atoi(argv[3]);
  unsigned int iterations = atoi(argv[4]);

  bool use_gpu = atoi(argv[5]) == 1;

  // float a[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  // float b[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
  // double *r = new double[4];
  // executeKernel(use_gpu, a, b, 4, r);
  // return 1;

  for(int size=min_size; size<=max_size; size+=step_size){
    vector<float> results;
    vector<float> buffer_a;
    vector<float> buffer_b;
    vector<float> buffer_c;

    float* array = generate(size, size);
    for(int i=0; i<iterations; i++){
      double *r = new double[4];
      executeKernel(use_gpu, array, array, size, r);
      results.push_back(r[0]);
      buffer_a.push_back(r[1]);
      buffer_b.push_back(r[2]);
      buffer_c.push_back(r[3]);
    }
    free(array);

    cout << size;

    double mean, stdev;

    mean = calcMean(buffer_a);
    stdev = calcStdev(buffer_a);
    cout << " Buffer A: avg=" << mean << "; stdev=" << stdev;

    mean = calcMean(buffer_b);
    stdev = calcStdev(buffer_b);
    cout << " Buffer B: avg=" << mean << "; stdev=" << stdev;

    mean = calcMean(buffer_c);
    stdev = calcStdev(buffer_c);
    cout << " Buffer C: avg=" << mean << "; stdev=" << stdev;

    mean = calcMean(results);
    stdev = calcStdev(results);
    cout << " Total: avg=" << mean << "; stdev=" << stdev << endl;
  }
}
