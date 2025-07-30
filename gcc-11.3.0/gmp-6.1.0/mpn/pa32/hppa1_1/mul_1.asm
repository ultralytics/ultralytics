dnl  HP-PA 1.1 mpn_mul_1 -- Multiply a limb vector with a limb and store the
dnl  result in a second limb vector.

dnl  Copyright 1992-1994, 2000-2002 Free Software Foundation, Inc.

dnl  This file is part of the GNU MP Library.
dnl
dnl  The GNU MP Library is free software; you can redistribute it and/or modify
dnl  it under the terms of either:
dnl
dnl    * the GNU Lesser General Public License as published by the Free
dnl      Software Foundation; either version 3 of the License, or (at your
dnl      option) any later version.
dnl
dnl  or
dnl
dnl    * the GNU General Public License as published by the Free Software
dnl      Foundation; either version 2 of the License, or (at your option) any
dnl      later version.
dnl
dnl  or both in parallel, as here.
dnl
dnl  The GNU MP Library is distributed in the hope that it will be useful, but
dnl  WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
dnl  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
dnl  for more details.
dnl
dnl  You should have received copies of the GNU General Public License and the
dnl  GNU Lesser General Public License along with the GNU MP Library.  If not,
dnl  see https://www.gnu.org/licenses/.

include(`../config.m4')

C INPUT PARAMETERS
C res_ptr	r26
C s1_ptr	r25
C size		r24
C s2_limb	r23

C This runs at 9 cycles/limb on a PA7000.  With the used instructions, it can
C not become faster due to data cache contention after a store.  On the PA7100
C it runs at 7 cycles/limb.

C We could use fldds to read two limbs at a time from the S1 array, and that
C could bring down the times to 8.5 and 6.5 cycles/limb for the PA7000 and
C PA7100, respectively.  We don't do that since it does not seem worth the
C (alignment) troubles...

C At least the PA7100 is rumored to be able to deal with cache-misses without
C stalling instruction issue.  If this is true, and the cache is actually also
C lockup-free, we should use a deeper software pipeline, and load from S1 very
C early!  (The loads and stores to -12(sp) will surely be in the cache.)

ASM_START()
PROLOGUE(mpn_mul_1)
C	.callinfo	frame=64,no_calls

	ldo		64(%r30),%r30
	fldws,ma	4(%r25),%fr5
	stw		%r23,-16(%r30)		C move s2_limb ...
	addib,=		-1,%r24,L(just_one_limb)
	 fldws		-16(%r30),%fr4		C ... into fr4
	add		%r0,%r0,%r0		C clear carry
	xmpyu		%fr4,%fr5,%fr6
	fldws,ma	4(%r25),%fr7
	fstds		%fr6,-16(%r30)
	xmpyu		%fr4,%fr7,%fr8
	ldw		-12(%r30),%r19		C least significant limb in product
	ldw		-16(%r30),%r28

	fstds		%fr8,-16(%r30)
	addib,=		-1,%r24,L(end)
	 ldw		-12(%r30),%r1

C Main loop
LDEF(loop)
	fldws,ma	4(%r25),%fr5
	stws,ma		%r19,4(%r26)
	addc		%r28,%r1,%r19
	xmpyu		%fr4,%fr5,%fr6
	ldw		-16(%r30),%r28
	fstds		%fr6,-16(%r30)
	addib,<>	-1,%r24,L(loop)
	 ldw		-12(%r30),%r1

LDEF(end)
	stws,ma		%r19,4(%r26)
	addc		%r28,%r1,%r19
	ldw		-16(%r30),%r28
	stws,ma		%r19,4(%r26)
	addc		%r0,%r28,%r28
	bv		0(%r2)
	 ldo		-64(%r30),%r30

LDEF(just_one_limb)
	xmpyu		%fr4,%fr5,%fr6
	fstds		%fr6,-16(%r30)
	ldw		-16(%r30),%r28
	ldo		-64(%r30),%r30
	bv		0(%r2)
	 fstws		%fr6R,0(%r26)
EPILOGUE()
