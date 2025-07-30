dnl  Copyright 1999, 2000, 2002, 2003 Free Software Foundation, Inc.

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


dnl  Optimizations:
dnl  * Avoid skip instructions
dnl  * Put carry-generating and carry-consuming insns consecutively
dnl  * Don't allocate any stack, "home" positions for parameters could be used.

include(`../config.m4')

define(`p0',`%r28')
define(`p1',`%r29')
define(`t32',`%r19')
define(`t0',`%r20')
define(`t1',`%r21')
define(`x',`%r22')
define(`m0',`%r23')
define(`m1',`%r24')

ifdef(`HAVE_ABI_2_0w',
`	.level	2.0w
',`	.level	2.0
')
PROLOGUE(mpn_umul_ppmm_r)
	ldo		128(%r30),%r30
ifdef(`HAVE_ABI_2_0w',
`	std		%r26,-64(%r30)
	std		%r25,-56(%r30)
	copy		%r24,%r31
',`
	depd		%r25,31,32,%r26
	std		%r26,-64(%r30)
	depd		%r23,31,32,%r24
	std		%r24,-56(%r30)
	ldw		-180(%r30),%r31
')

	fldd		-64(%r30),%fr4
	fldd		-56(%r30),%fr5

	xmpyu		%fr5R,%fr4R,%fr6
	fstd		%fr6,-128(%r30)
	xmpyu		%fr5R,%fr4L,%fr7
	fstd		%fr7,-120(%r30)
	xmpyu		%fr5L,%fr4R,%fr8
	fstd		%fr8,-112(%r30)
	xmpyu		%fr5L,%fr4L,%fr9
	fstd		%fr9,-104(%r30)

	depdi,z		1,31,1,t32		C t32 = 2^32

	ldd		-128(%r30),p0		C lo = low 64 bit of product
	ldd		-120(%r30),m0		C m0 = mid0 64 bit of product
	ldd		-112(%r30),m1		C m1 = mid1 64 bit of product
	ldd		-104(%r30),p1		C hi = high 64 bit of product

	add,l,*nuv	m0,m1,x			C x = m1+m0
	 add,l		t32,p1,p1		C propagate carry to mid of p1
	depd,z		x,31,32,t0		C lo32(m1+m0)
	add		t0,p0,p0
	extrd,u		x,31,32,t1		C hi32(m1+m0)
	add,dc		t1,p1,p1

	std		p0,0(%r31)		C store low half of product
ifdef(`HAVE_ABI_2_0w',
`	copy		p1,%r28			C return val in %r28
',`	extrd,u		p1,31,32,%r28		C return val in %r28,%r29
')
	bve		(%r2)
	ldo		-128(%r30),%r30
EPILOGUE(mpn_umul_ppmm_r)

