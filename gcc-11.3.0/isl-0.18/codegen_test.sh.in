#!/bin/sh

EXEEXT=@EXEEXT@
srcdir=@srcdir@

failed=0

for i in $srcdir/test_inputs/codegen/*.st \
		$srcdir/test_inputs/codegen/cloog/*.st; do
	echo $i;
	base=`basename $i .st`
	test=test-$base.c
	dir=`dirname $i`
	ref=$dir/$base.c
	(./isl_codegen$EXEEXT < $i > $test &&
	 diff -uw $ref $test && rm $test) || failed=1
done
for i in $srcdir/test_inputs/codegen/*.in \
		$srcdir/test_inputs/codegen/omega/*.in \
		$srcdir/test_inputs/codegen/pldi2012/*.in; do
	echo $i;
	base=`basename $i .in`
	test=test-$base.c
	dir=`dirname $i`
	ref=$dir/$base.c
	(./isl_codegen$EXEEXT < $i > $test &&
	 diff -uw $ref $test && rm $test) || failed=1
done

test $failed -eq 0 || exit
