dnl  HP-PA 2.0 64-bit mpn_sqr_diagonal.

dnl  Copyright 2001-2003 Free Software Foundation, Inc.

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


dnl  This code runs at 7.25 cycles/limb on PA8000 and 7.75 cycles/limb on
dnl  PA8500.  The cache would saturate at 5 cycles/limb, so there is some room
dnl  for optimization.

include(`../config.m4')

C INPUT PARAMETERS
define(`rp',`%r26')
define(`up',`%r25')
define(`n',`%r24')

define(`p00',`%r28')
define(`p32',`%r29')
define(`p64',`%r31')
define(`t0',`%r19')
define(`t1',`%r20')

ifdef(`HAVE_ABI_2_0w',
`	.level	2.0w
',`	.level	2.0
')
PROLOGUE(mpn_sqr_diagonal)
	ldo		128(%r30),%r30

	fldds,ma	8(up),%fr8
	addib,=		-1,n,L(end1)
	nop
	fldds,ma	8(up),%fr4
	xmpyu		%fr8l,%fr8r,%fr10
	fstd		%fr10,-120(%r30)
	xmpyu		%fr8r,%fr8r,%fr9
	fstd		%fr9,0(rp)
	xmpyu		%fr8l,%fr8l,%fr11
	fstd		%fr11,8(rp)
	addib,=		-1,n,L(end2)
	ldo		16(rp),rp

LDEF(loop)
	fldds,ma	8(up),%fr8		C load next up limb
	xmpyu		%fr4l,%fr4r,%fr6
	fstd		%fr6,-128(%r30)
	xmpyu		%fr4r,%fr4r,%fr5	C multiply in fp regs
	fstd		%fr5,0(rp)
	xmpyu		%fr4l,%fr4l,%fr7
	fstd		%fr7,8(rp)
	ldd		-120(%r30),p32
	ldd		-16(rp),p00		C accumulate in int regs
	ldd		-8(rp),p64
	depd,z		p32,30,31,t0
	add		t0,p00,p00
	std		p00,-16(rp)
	extrd,u		p32,32,33,t1
	add,dc		t1,p64,p64
	std		p64,-8(rp)
	addib,=		-1,n,L(exit)
	ldo		16(rp),rp

	fldds,ma	8(up),%fr4
	xmpyu		%fr8l,%fr8r,%fr10
	fstd		%fr10,-120(%r30)
	xmpyu		%fr8r,%fr8r,%fr9
	fstd		%fr9,0(rp)
	xmpyu		%fr8l,%fr8l,%fr11
	fstd		%fr11,8(rp)
	ldd		-128(%r30),p32
	ldd		-16(rp),p00
	ldd		-8(rp),p64
	depd,z		p32,30,31,t0
	add		t0,p00,p00
	std		p00,-16(rp)
	extrd,u		p32,32,33,t1
	add,dc		t1,p64,p64
	std		p64,-8(rp)
	addib,<>	-1,n,L(loop)
	ldo		16(rp),rp

LDEF(end2)
	xmpyu		%fr4l,%fr4r,%fr6
	fstd		%fr6,-128(%r30)
	xmpyu		%fr4r,%fr4r,%fr5
	fstd		%fr5,0(rp)
	xmpyu		%fr4l,%fr4l,%fr7
	fstd		%fr7,8(rp)
	ldd		-120(%r30),p32
	ldd		-16(rp),p00
	ldd		-8(rp),p64
	depd,z		p32,30,31,t0
	add		t0,p00,p00
	std		p00,-16(rp)
	extrd,u		p32,32,33,t1
	add,dc		t1,p64,p64
	std		p64,-8(rp)
	ldo		16(rp),rp
	ldd		-128(%r30),p32
	ldd		-16(rp),p00
	ldd		-8(rp),p64
	depd,z		p32,30,31,t0
	add		t0,p00,p00
	std		p00,-16(rp)
	extrd,u		p32,32,33,t1
	add,dc		t1,p64,p64
	std		p64,-8(rp)
	bve		(%r2)
	ldo		-128(%r30),%r30

LDEF(exit)
	xmpyu		%fr8l,%fr8r,%fr10
	fstd		%fr10,-120(%r30)
	xmpyu		%fr8r,%fr8r,%fr9
	fstd		%fr9,0(rp)
	xmpyu		%fr8l,%fr8l,%fr11
	fstd		%fr11,8(rp)
	ldd		-128(%r30),p32
	ldd		-16(rp),p00
	ldd		-8(rp),p64
	depd,z		p32,31,32,t0
	add		t0,p00,p00
	extrd,u		p32,31,32,t1
	add,dc		t1,p64,p64
	add		t0,p00,p00
	add,dc		t1,p64,p64
	std		p00,-16(rp)
	std		p64,-8(rp)
	ldo		16(rp),rp
	ldd		-120(%r30),p32
	ldd		-16(rp),p00
	ldd		-8(rp),p64
	depd,z		p32,31,32,t0
	add		t0,p00,p00
	extrd,u		p32,31,32,t1
	add,dc		t1,p64,p64
	add		t0,p00,p00
	add,dc		t1,p64,p64
	std		p00,-16(rp)
	std		p64,-8(rp)
	bve		(%r2)
	ldo		-128(%r30),%r30

LDEF(end1)
	xmpyu		%fr8l,%fr8r,%fr10
	fstd		%fr10,-128(%r30)
	xmpyu		%fr8r,%fr8r,%fr9
	fstd		%fr9,0(rp)
	xmpyu		%fr8l,%fr8l,%fr11
	fstd		%fr11,8(rp)
	ldo		16(rp),rp
	ldd		-128(%r30),p32
	ldd		-16(rp),p00
	ldd		-8(rp),p64
	depd,z		p32,31,32,t0
	add		t0,p00,p00
	extrd,u		p32,31,32,t1
	add,dc		t1,p64,p64
	add		t0,p00,p00
	add,dc		t1,p64,p64
	std		p00,-16(rp)
	std		p64,-8(rp)
	bve		(%r2)
	ldo		-128(%r30),%r30
EPILOGUE(mpn_sqr_diagonal)
