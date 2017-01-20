typedef struct This_s{
   __global float *a;
   __global float *b;
   __global float *c;
   int size;
}This;

__kernel void matrix_multiplication(__global float *a,
                       __global float *b,
                       __global float *c,
                       const unsigned int size)
{
This thisStruct;
This* this=&thisStruct;
this->a = a;
this->size = size;
this->b = b;
this->c = c;
{

    int y = get_global_id(0);
    int x = get_global_id(1);
    float sum = 0;
    for(int i = 0; i < this->size; i++){
       sum += this->a[y * this->size + i] * this->b[x * this->size + i];
    }
    this->c[y * this->size + x] = sum;
}
}
