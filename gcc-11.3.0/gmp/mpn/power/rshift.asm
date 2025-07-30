dnl  IBM POWER mpn_rshift -- Shift a number right.

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
dnl  s_ptr	r4
dnl  size	r5
dnl  cnt	r6

include(`../config.m4')

ASM_START()
PROLOGUE(mpn_rshift)
	sfi	8,6,32
	mtctr	5		C put limb count in CTR loop register
	l	0,0(4)		C read least significant limb
	ai	9,3,-4		C adjust res_ptr since it's offset in the stu:s
	sle	3,0,8		C compute carry limb, and init MQ register
	bdz	Lend2		C if just one limb, skip loop
	lu	0,4(4)		C read 2:nd least significant limb
	sleq	7,0,8		C compute least significant limb of result
	bdz	Lend		C if just two limb, skip loop
Loop:	lu	0,4(4)		C load next higher limb
	stu	7,4(9)		C store previous result during read latency
	sleq	7,0,8		C compute result limb
	bdn	Loop		C loop back until CTR is zero
Lend:	stu	7,4(9)		C store 2:nd most significant limb
Lend2:	sre	7,0,6		C compute most significant limb
	st	7,4(9)		C store it
	br
EPILOGUE(mpn_rshift)
