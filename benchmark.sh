echo $1
for i in `seq 1 12`;
do
  product=`expr $i \* 500`
  echo $product
  ./alg $1 $product $product $product
done
