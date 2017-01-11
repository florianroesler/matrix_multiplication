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

int main( int argc, char** argv)
{
  unsigned int min_size = atoi(argv[1]);
  unsigned int max_size = atoi(argv[2]);
  unsigned int step_size = atoi(argv[3]);
  unsigned int iterations = atoi(argv[4]);

  bool use_gpu = atoi(argv[5]) == 1;

  for(int size=min_size; size<=max_size; size+=step_size){
    vector<double> results;
    double* array = generate(size, size);
    for(int i=0; i<iterations; i++){
      double r = executeKernel(use_gpu, array, array, size);
      results.push_back(r);
    }
    free(array);
    double sum = accumulate(results.begin(), results.end(), 0.0);
    double mean = sum / results.size();
    double sq_sum = inner_product(results.begin(), results.end(), results.begin(), 0.0);
    double stdev = sqrt(sq_sum / results.size() - mean * mean);

    cout << size << ": avg=" << mean << "; stdev=" << stdev << endl;
  }


}
