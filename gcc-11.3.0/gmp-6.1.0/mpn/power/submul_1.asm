dnl  IBM POWER mpn_submul_1 -- Multiply a limb vector with a limb and subtract
dnl  the result from a second limb vector.

dnl  Copyright 1992, 1994, 1999-2001 Free Software Foundation, Inc.

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


dnl  INPUT PARAMETERS
dnl  res_ptr	r3
dnl  s1_ptr	r4
dnl  size	r5
dnl  s2_limb	r6

dnl  The POWER architecture has no unsigned 32x32->64 bit multiplication
dnl  instruction.  To obtain that operation, we have to use the 32x32->64
dnl  signed multiplication instruction, and add the appropriate compensation to
dnl  the high limb of the result.  We add the multiplicand if the multiplier
dnl  has its most significant bit set, and we add the multiplier if the
dnl  multiplicand has its most significant bit set.  We need to preserve the
dnl  carry flag between each iteration, so we have to compute the compensation
dnl  carefully (the natural, srai+and doesn't work).  Since all POWER can
dnl  branch in zero cycles, we use conditional branches for the compensation.

include(`../config.m4')

ASM_START()
PROLOGUE(mpn_submul_1)
	cal	3,-4(3)
	l	0,0(4)
	cmpi	0,6,0
	mtctr	5
	mul	9,0,6
	srai	7,0,31
	and	7,7,6
	mfmq	11
	cax	9,9,7
	l	7,4(3)
	sf	8,11,7		C add res_limb
	a	11,8,11		C invert cy (r11 is junk)
	blt	Lneg
Lpos:	bdz	Lend

Lploop:	lu	0,4(4)
	stu	8,4(3)
	cmpi	0,0,0
	mul	10,0,6
	mfmq	0
	ae	11,0,9		C low limb + old_cy_limb + old cy
	l	7,4(3)
	aze	10,10		C propagate cy to new cy_limb
	sf	8,11,7		C add res_limb
	a	11,8,11		C invert cy (r11 is junk)
	bge	Lp0
	cax	10,10,6		C adjust high limb for negative limb from s1
Lp0:	bdz	Lend0
	lu	0,4(4)
	stu	8,4(3)
	cmpi	0,0,0
	mul	9,0,6
	mfmq	0
	ae	11,0,10
	l	7,4(3)
	aze	9,9
	sf	8,11,7
	a	11,8,11		C invert cy (r11 is junk)
	bge	Lp1
	cax	9,9,6		C adjust high limb for negative limb from s1
Lp1:	bdn	Lploop

	b	Lend

Lneg:	cax	9,9,0
	bdz	Lend
Lnloop:	lu	0,4(4)
	stu	8,4(3)
	cmpi	0,0,0
	mul	10,0,6
	mfmq	7
	ae	11,7,9
	l	7,4(3)
	ae	10,10,0		C propagate cy to new cy_limb
	sf	8,11,7		C add res_limb
	a	11,8,11		C invert cy (r11 is junk)
	bge	Ln0
	cax	10,10,6		C adjust high limb for negative limb from s1
Ln0:	bdz	Lend0
	lu	0,4(4)
	stu	8,4(3)
	cmpi	0,0,0
	mul	9,0,6
	mfmq	7
	ae	11,7,10
	l	7,4(3)
	ae	9,9,0		C propagate cy to new cy_limb
	sf	8,11,7		C add res_limb
	a	11,8,11		C invert cy (r11 is junk)
	bge	Ln1
	cax	9,9,6		C adjust high limb for negative limb from s1
Ln1:	bdn	Lnloop
	b	Lend

Lend0:	cal	9,0(10)
Lend:	st	8,4(3)
	aze	3,9
	br
EPILOGUE(mpn_submul_1)
