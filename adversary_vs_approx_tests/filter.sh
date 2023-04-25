for I in mul8u_*.log
do
    BASE=`basename  $I .log`
    echo $BASE
    cat $I | grep "robust accuracy after SQUARE" | cut -d" " -f5 > ${BASE}.txt
done
