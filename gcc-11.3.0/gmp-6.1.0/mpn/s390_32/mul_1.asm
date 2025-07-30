dnl  S/390 mpn_mul_1 -- Multiply a limb vector with a limb and store the
dnl  result in a second limb vector.

dnl  Copyright 2001 Free Software Foundation, Inc.

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

define(`rp',2)
define(`up',3)
define(`n',4)
define(`vlimb',5)
define(`cylimb',7)

ASM_START()
PROLOGUE(mpn_mul_1)
	stm	6,7,24(15)
	slr	cylimb,cylimb	# clear cylimb
	ltr	vlimb,vlimb
	jnl	.Loopp

.Loopn:	l	1,0(up)		# load from u
	lr	6,1		#
	mr	0,vlimb		# multiply signed
	alr	0,6		# add vlimb to phi
	sra	6,31		# make mask
	nr	6,vlimb		# 0 or vlimb
	alr	0,6		# conditionally add vlimb to phi
	alr	1,cylimb	# add carry limb to plo
	brc	8+4,+8		# branch if not carry
	ahi	0,1		# increment phi
	lr	cylimb,0	# new cylimb
	st	1,0(rp)		# store
	la	up,4(,up)
	la	rp,4(,rp)
	brct	n,.Loopn

	lr	2,cylimb
	lm	6,7,24(15)
	br	14

.Loopp:	l	1,0(up)		# load from u
	lr	6,1		#
	mr	0,vlimb		# multiply signed
	sra	6,31		# make mask
	nr	6,vlimb		# 0 or vlimb
	alr	0,6		# conditionally add vlimb to phi
	alr	1,cylimb	# add carry limb to plo
	brc	8+4,+8		# branch if not carry
	ahi	0,1		# increment phi
	lr	cylimb,0	# new cylimb
	st	1,0(rp)		# store
	la	up,4(,up)
	la	rp,4(,rp)
	brct	n,.Loopp

	lr	2,cylimb
	lm	6,7,24(15)
	br	14
EPILOGUE(mpn_mul_1)
