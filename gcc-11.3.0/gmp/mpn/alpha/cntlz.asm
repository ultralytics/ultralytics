dnl  Alpha auxiliary for longlong.h's count_leading_zeros

dnl  Copyright 1997, 2000, 2002 Free Software Foundation, Inc.

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


ASM_START()
EXTERN(__clz_tab)
PROLOGUE(mpn_count_leading_zeros,gp)
	cmpbge	r31,  r16, r1
	LEA(r3,__clz_tab)
	sra	r1,   1,   r1
	xor	r1,   127, r1
	srl	r16,  1,   r16
	addq	r1,   r3,  r1
	ldq_u	r0,   0(r1)
	lda	r2,   64
	extbl	r0,   r1,   r0
	s8subl	r0,   8,    r0
	srl	r16,  r0,   r16
	addq	r16,  r3,   r16
	ldq_u	r1,   0(r16)
	extbl	r1,   r16,  r1
	subq	r2,   r1,   r2
	subq	r2,   r0,   r0
	ret	r31,  (r26),1
EPILOGUE(mpn_count_leading_zeros)
ASM_END()
