#!/bin/sh

EXEEXT=@EXEEXT@
srcdir=@srcdir@

BOUND_TESTS="\
	basicLinear2.pwqp \
	basicLinear.pwqp \
	basicTestParameterPosNeg.pwqp \
	basicTest.pwqp \
	devos.pwqp \
	equality1.pwqp \
	equality2.pwqp \
	equality3.pwqp \
	equality4.pwqp \
	equality5.pwqp \
	faddeev.pwqp \
	linearExample.pwqp \
	neg.pwqp \
	philippe3vars3pars.pwqp \
	philippe3vars.pwqp \
	philippeNeg.pwqp \
	philippePolynomialCoeff1P.pwqp \
	philippePolynomialCoeff.pwqp \
	philippe.pwqp \
	product.pwqp \
	split.pwqp \
	test3Deg3Var.pwqp \
	toplas.pwqp \
	unexpanded.pwqp"

for i in $BOUND_TESTS; do
	echo $i;
	./isl_bound$EXEEXT -T --bound=bernstein < $srcdir/test_inputs/$i || exit
	./isl_bound$EXEEXT -T --bound=range < $srcdir/test_inputs/$i || exit
done
