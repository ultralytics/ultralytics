#!/bin/sh

EXEEXT=@EXEEXT@
srcdir=@srcdir@

PIP_TESTS="\
	boulet.pip \
	brisebarre.pip \
	cg1.pip \
	esced.pip \
	ex2.pip \
	ex.pip \
	exist.pip \
	exist2.pip \
	fimmel.pip \
	max.pip \
	negative.pip \
	seghir-vd.pip \
	small.pip \
	sor1d.pip \
	square.pip \
	sven.pip \
	tobi.pip"

for i in $PIP_TESTS; do
	echo $i;
	./isl_pip$EXEEXT --format=set --context=gbr -T < $srcdir/test_inputs/$i || exit
	./isl_pip$EXEEXT --format=set --context=lexmin -T < $srcdir/test_inputs/$i || exit
	./isl_pip$EXEEXT --format=affine --context=gbr -T < $srcdir/test_inputs/$i || exit
	./isl_pip$EXEEXT --format=affine --context=lexmin -T < $srcdir/test_inputs/$i || exit
done
