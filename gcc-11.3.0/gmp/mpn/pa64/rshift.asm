dnl  HP-PA 2.0 mpn_rshift -- Right shift.

dnl  Copyright 1997, 2000, 2002, 2003 Free Software Foundation, Inc.

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


dnl  This runs at 1.5 cycles/limb on PA8000 and 1.0 cycles/limb on PA8500.

include(`../config.m4')

dnl  INPUT PARAMETERS
define(`rp',`%r26')
define(`up',`%r25')
define(`n',`%r24')
define(`cnt',`%r23')

ifdef(`HAVE_ABI_2_0w',
`       .level  2.0w
',`     .level  2.0
')
PROLOGUE(mpn_rshift)
	mtsar		cnt
	ldd		0(up), %r21
	addib,=		-1, n, L(end)
	shrpd		%r21, %r0, %sar, %r29	C compute carry out limb
	depw,z		n, 31, 3, %r28		C r28 = (size & 7)
	sub		%r0, n, %r22
	depw,z		%r22, 28, 3, %r22	C r22 = 8 * (-size & 7)
	sub		up, %r22, up		C offset up
	blr		%r28, %r0		C branch into jump table
	sub		rp, %r22, rp		C offset rp
	b		L(0)
	nop
	b		L(1)
	copy		%r21, %r20
	b		L(2)
	nop
	b		L(3)
	copy		%r21, %r20
	b		L(4)
	nop
	b		L(5)
	copy		%r21, %r20
	b		L(6)
	nop
	b		L(7)
	copy		%r21, %r20

LDEF(loop)
LDEF(0)	ldd		8(up), %r20
	shrpd		%r20, %r21, %sar, %r21
	std		%r21, 0(rp)
LDEF(7)	ldd		16(up), %r21
	shrpd		%r21, %r20, %sar, %r20
	std		%r20, 8(rp)
LDEF(6)	ldd		24(up), %r20
	shrpd		%r20, %r21, %sar, %r21
	std		%r21, 16(rp)
LDEF(5)	ldd		32(up), %r21
	shrpd		%r21, %r20, %sar, %r20
	std		%r20, 24(rp)
LDEF(4)	ldd		40(up), %r20
	shrpd		%r20, %r21, %sar, %r21
	std		%r21, 32(rp)
LDEF(3)	ldd		48(up), %r21
	shrpd		%r21, %r20, %sar, %r20
	std		%r20, 40(rp)
LDEF(2)	ldd		56(up), %r20
	shrpd		%r20, %r21, %sar, %r21
	std		%r21, 48(rp)
LDEF(1)	ldd		64(up), %r21
	ldo		64(up), up
	shrpd		%r21, %r20, %sar, %r20
	std		%r20, 56(rp)
	addib,>		-8, n, L(loop)
	ldo		64(rp), rp

LDEF(end)
	shrpd		%r0, %r21, %sar, %r21
	std		%r21, 0(rp)
	bve		(%r2)
ifdef(`HAVE_ABI_2_0w',
`	copy		%r29,%r28
',`	extrd,u		%r29, 31, 32, %r28
')
EPILOGUE(mpn_rshift)
