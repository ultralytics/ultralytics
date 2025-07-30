dnl  HP-PA 7100/7200 mpn_submul_1 -- Multiply a limb vector with a limb and
dnl  subtract the result from a second limb vector.

dnl  Copyright 1995, 2000-2003 Free Software Foundation, Inc.

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
define(`res_ptr',`%r26')
define(`s1_ptr',`%r25')
define(`size_param',`%r24')
define(`s2_limb',`%r23')

define(`cylimb',`%r28')
define(`s0',`%r19')
define(`s1',`%r20')
define(`s2',`%r3')
define(`s3',`%r4')
define(`lo0',`%r21')
define(`lo1',`%r5')
define(`lo2',`%r6')
define(`lo3',`%r7')
define(`hi0',`%r22')
define(`hi1',`%r23')				C safe to reuse
define(`hi2',`%r29')
define(`hi3',`%r1')

ASM_START()
PROLOGUE(mpn_submul_1)
C	.callinfo	frame=128,no_calls

	ldo	128(%r30),%r30
	stws	s2_limb,-16(%r30)
	add	 %r0,%r0,cylimb			C clear cy and cylimb
	addib,<	-4,size_param,L(few_limbs)
	fldws	-16(%r30),%fr31R

	ldo	-112(%r30),%r31
	stw	%r3,-96(%r30)
	stw	%r4,-92(%r30)
	stw	%r5,-88(%r30)
	stw	%r6,-84(%r30)
	stw	%r7,-80(%r30)

	bb,>=,n	 s1_ptr,29,L(0)

	fldws,ma 4(s1_ptr),%fr4
	ldws	 0(res_ptr),s0
	xmpyu	 %fr4,%fr31R,%fr5
	fstds	 %fr5,-16(%r31)
	ldws	-16(%r31),cylimb
	ldws	-12(%r31),lo0
	sub	 s0,lo0,s0
	add	 s0,lo0,%r0			C invert cy
	addib,< -1,size_param,L(few_limbs)
	stws,ma	 s0,4(res_ptr)

C start software pipeline ----------------------------------------------------
LDEF(0)
	fldds,ma 8(s1_ptr),%fr4
	fldds,ma 8(s1_ptr),%fr8

	xmpyu	 %fr4L,%fr31R,%fr5
	xmpyu	 %fr4R,%fr31R,%fr6
	xmpyu	 %fr8L,%fr31R,%fr9
	xmpyu	 %fr8R,%fr31R,%fr10

	fstds	 %fr5,-16(%r31)
	fstds	 %fr6,-8(%r31)
	fstds	 %fr9,0(%r31)
	fstds	 %fr10,8(%r31)

	ldws   -16(%r31),hi0
	ldws   -12(%r31),lo0
	ldws	-8(%r31),hi1
	ldws	-4(%r31),lo1
	ldws	 0(%r31),hi2
	ldws	 4(%r31),lo2
	ldws	 8(%r31),hi3
	ldws	12(%r31),lo3

	addc	 lo0,cylimb,lo0
	addc	 lo1,hi0,lo1
	addc	 lo2,hi1,lo2
	addc	 lo3,hi2,lo3

	addib,<	 -4,size_param,L(end)
	addc	 %r0,hi3,cylimb			C propagate carry into cylimb
C main loop ------------------------------------------------------------------
LDEF(loop)
	fldds,ma 8(s1_ptr),%fr4
	fldds,ma 8(s1_ptr),%fr8

	ldws	 0(res_ptr),s0
	xmpyu	 %fr4L,%fr31R,%fr5
	ldws	 4(res_ptr),s1
	xmpyu	 %fr4R,%fr31R,%fr6
	ldws	 8(res_ptr),s2
	xmpyu	 %fr8L,%fr31R,%fr9
	ldws	12(res_ptr),s3
	xmpyu	 %fr8R,%fr31R,%fr10

	fstds	 %fr5,-16(%r31)
	sub	 s0,lo0,s0
	fstds	 %fr6,-8(%r31)
	subb	 s1,lo1,s1
	fstds	 %fr9,0(%r31)
	subb	 s2,lo2,s2
	fstds	 %fr10,8(%r31)
	subb	 s3,lo3,s3
	subb	 %r0,%r0,lo0			C these two insns ...
	add	 lo0,lo0,%r0			C ... just invert cy

	ldws   -16(%r31),hi0
	ldws   -12(%r31),lo0
	ldws	-8(%r31),hi1
	ldws	-4(%r31),lo1
	ldws	 0(%r31),hi2
	ldws	 4(%r31),lo2
	ldws	 8(%r31),hi3
	ldws	12(%r31),lo3

	addc	 lo0,cylimb,lo0
	stws,ma	 s0,4(res_ptr)
	addc	 lo1,hi0,lo1
	stws,ma	 s1,4(res_ptr)
	addc	 lo2,hi1,lo2
	stws,ma	 s2,4(res_ptr)
	addc	 lo3,hi2,lo3
	stws,ma	 s3,4(res_ptr)

	addib,>= -4,size_param,L(loop)
	addc	 %r0,hi3,cylimb			C propagate carry into cylimb
C finish software pipeline ---------------------------------------------------
LDEF(end)
	ldws	 0(res_ptr),s0
	ldws	 4(res_ptr),s1
	ldws	 8(res_ptr),s2
	ldws	12(res_ptr),s3

	sub	 s0,lo0,s0
	stws,ma	 s0,4(res_ptr)
	subb	 s1,lo1,s1
	stws,ma	 s1,4(res_ptr)
	subb	 s2,lo2,s2
	stws,ma	 s2,4(res_ptr)
	subb	 s3,lo3,s3
	stws,ma	 s3,4(res_ptr)
	subb	 %r0,%r0,lo0			C these two insns ...
	add	 lo0,lo0,%r0			C ... invert cy

C restore callee-saves registers ---------------------------------------------
	ldw	-96(%r30),%r3
	ldw	-92(%r30),%r4
	ldw	-88(%r30),%r5
	ldw	-84(%r30),%r6
	ldw	-80(%r30),%r7

LDEF(few_limbs)
	addib,=,n 4,size_param,L(ret)

LDEF(loop2)
	fldws,ma 4(s1_ptr),%fr4
	ldws	 0(res_ptr),s0
	xmpyu	 %fr4,%fr31R,%fr5
	fstds	 %fr5,-16(%r30)
	ldws	-16(%r30),hi0
	ldws	-12(%r30),lo0
	addc	 lo0,cylimb,lo0
	addc	 %r0,hi0,cylimb
	sub	 s0,lo0,s0
	add	 s0,lo0,%r0			C invert cy
	stws,ma	 s0,4(res_ptr)
	addib,<> -1,size_param,L(loop2)
	nop

LDEF(ret)
	addc	 %r0,cylimb,cylimb
	bv	 0(%r2)
	ldo	 -128(%r30),%r30
EPILOGUE(mpn_submul_1)
