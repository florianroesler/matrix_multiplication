

float* generate(unsigned int height, unsigned int width){
  int i, j;
  float* matrix = new float[height * width];
  for(i = 0; i < height; i++)
  {
    for(j = 0; j < width; j++)
    {
      matrix[i * width + j] = randMToN(1, 3);
    }
  }
  return matrix;
}
