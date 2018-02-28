#!/bin/bash

awk -F ' ' 'BEGIN{sum=0;count=0}{sum+=1;if ($NF==$(NF-1)){count+=1}}END{print "count:"count",sum:"sum",total:"count/sum}' testRet.txt
awk -F ' ' 'BEGIN{sum=0;count=0}{if ($(NF-1)=="other"){sum+=1;if ($NF==$(NF-1)){count+=1}}}END{print "count:"count",sum:"sum",other:"count/sum}' testRet.txt
awk -F ' ' 'BEGIN{sum=0;count=0}{if ($(NF-1)!="other"){sum+=1;if ($NF==$(NF-1)){count+=1}}}END{print "count:"count",sum:"sum",business:"count/sum}' testRet.txt
