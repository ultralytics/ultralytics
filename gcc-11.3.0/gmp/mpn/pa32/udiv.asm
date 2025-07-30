dnl  HP-PA  __udiv_qrnnd division support, used from longlong.h.
dnl  This version runs fast on pre-PA7000 CPUs.

dnl  Copyright 1993, 1994, 2000-2002 Free Software Foundation, Inc.

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
C rem_ptr	gr26
C n1		gr25
C n0		gr24
C d		gr23

C The code size is a bit excessive.  We could merge the last two ds;addc
C sequences by simply moving the "bb,< Odd" instruction down.  The only
C trouble is the FFFFFFFF code that would need some hacking.

ASM_START()
PROLOGUE(mpn_udiv_qrnnd)
	comb,<		%r23,0,L(largedivisor)
	 sub		%r0,%r23,%r1		C clear cy as side-effect
	ds		%r0,%r1,%r0
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r23,%r25
	addc		%r24,%r24,%r28
	ds		%r25,%r23,%r25
	comclr,>=	%r25,%r0,%r0
	addl		%r25,%r23,%r25
	stws		%r25,0(0,%r26)
	bv		0(%r2)
	 addc		%r28,%r28,%r28

LDEF(largedivisor)
	extru		%r24,31,1,%r19		C r19 = n0 & 1
	bb,<		%r23,31,L(odd)
	 extru		%r23,30,31,%r22		C r22 = d >> 1
	shd		%r25,%r24,1,%r24	C r24 = new n0
	extru		%r25,30,31,%r25		C r25 = new n1
	sub		%r0,%r22,%r21
	ds		%r0,%r21,%r0
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	comclr,>=	%r25,%r0,%r0
	addl		%r25,%r22,%r25
	sh1addl		%r25,%r19,%r25
	stws		%r25,0(0,%r26)
	bv		0(%r2)
	 addc		%r24,%r24,%r28

LDEF(odd)
	addib,sv,n	1,%r22,L(FFFFFFFF)	C r22 = (d / 2 + 1)
	shd		%r25,%r24,1,%r24	C r24 = new n0
	extru		%r25,30,31,%r25		C r25 = new n1
	sub		%r0,%r22,%r21
	ds		%r0,%r21,%r0
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r24
	ds		%r25,%r22,%r25
	addc		%r24,%r24,%r28
	comclr,>=	%r25,%r0,%r0
	addl		%r25,%r22,%r25
	sh1addl		%r25,%r19,%r25
C We have computed (n1,,n0) / (d + 1), q' = r28, r' = r25
	add,nuv		%r28,%r25,%r25
	addl		%r25,%r1,%r25
	addc		%r0,%r28,%r28
	sub,<<		%r25,%r23,%r0
	addl		%r25,%r1,%r25
	stws		%r25,0(0,%r26)
	bv		0(%r2)
	 addc		%r0,%r28,%r28

C This is just a special case of the code above.
C We come here when d == 0xFFFFFFFF
LDEF(FFFFFFFF)
	add,uv		%r25,%r24,%r24
	sub,<<		%r24,%r23,%r0
	ldo		1(%r24),%r24
	stws		%r24,0(0,%r26)
	bv		0(%r2)
	 addc		%r0,%r25,%r28
EPILOGUE()
