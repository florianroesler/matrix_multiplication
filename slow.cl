__kernel void matrix_multiplication(__global float *a,
                       __global float *b,
                       __global float *c,
                       const unsigned int size)
{
    int y = get_global_id(0);
    int x = get_global_id(1);
    float sum = 0;
    for(int i = 0; i < size; i++){
       sum += a[y * size + i] * b[x * size + i];
    }
    c[y * size + x] = sum;
}
