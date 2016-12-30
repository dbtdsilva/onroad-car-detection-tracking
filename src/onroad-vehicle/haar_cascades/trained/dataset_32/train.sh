#!/bin/bash

# $1 for extra arguments in the opencv_createsamples
find positive -iname *.png -exec echo \{\} 1 0 0 16 16 \; > cars.info
find negative -iname *.png > bg.txt
opencv_createsamples -info cars.info -num 3425 -w 16 -h 16 -vec car.vec $1
