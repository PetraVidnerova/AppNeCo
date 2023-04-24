for I in ../../axmul_8x8/*
do
    NAME=`basename $I`
    echo $NAME
    for I in `seq 1 10`
    do
	python eval_tf2_approx.py $NAME >> ${NAME}_robust_accuracy.log
    done
done
