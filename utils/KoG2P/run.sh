#!/bin/bash

touch word2phon2.txt

while read i;do
	#echo $i
	python3 g2p.py \'${i}\' >> word2phon2.txt
done < words.txt
