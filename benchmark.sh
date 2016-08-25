for i in `seq 1 12`;
do
  product=`expr $i \* 500`
  echo $product
  ./alg $product $product $product
done
