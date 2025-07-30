dnl  HP-PA 2.0 mpn_add_n, mpn_sub_n

dnl  Copyright 1997, 2000, 2002, 2003, 2009, 2010 Free Software Foundation,
dnl  Inc.

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


dnl  This runs at 2 cycles/limb on PA8000 and 1.6875 cycles/limb on PA8500.  It
dnl  should be possible to reach the cache bandwidth 1.5 cycles/limb at least
dnl  with PA8500.  The problem now is stalling of the first ADD,DC after LDO,
dnl  where the processor gets confused about where carry comes from.

include(`../config.m4')

dnl INPUT PARAMETERS
define(`rp',`%r26')
define(`up',`%r25')
define(`vp',`%r24')
define(`n',`%r23')

ifdef(`OPERATION_add_n', `
	define(ADCSBC,	      `add,dc')
	define(INITCY,	      `addi -1,%r22,%r0')
	define(func,	      mpn_add_n)
	define(func_nc,	      mpn_add_nc)')
ifdef(`OPERATION_sub_n', `
	define(ADCSBC,	      `sub,db')
	define(INITCY,	      `subi 0,%r22,%r0')
	define(func,	      mpn_sub_n)
	define(func_nc,	      mpn_sub_nc)')

MULFUNC_PROLOGUE(mpn_add_n mpn_add_nc mpn_sub_n mpn_sub_nc)

ifdef(`HAVE_ABI_2_0w',
`       .level  2.0w
',`     .level  2.0
')
PROLOGUE(func_nc)
ifdef(`HAVE_ABI_2_0w',
`	b		L(com)
	nop
',`	b		L(com)
	ldw		-52(%r30), %r22
')
EPILOGUE()
PROLOGUE(func)
	ldi		0, %r22
LDEF(com)
	sub		%r0, n, %r21
	depw,z		%r21, 30, 3, %r28	C r28 = 2 * (-n & 7)
	depw,z		%r21, 28, 3, %r21	C r21 = 8 * (-n & 7)
	sub		up, %r21, up		C offset up
	sub		vp, %r21, vp		C offset vp
	sub		rp, %r21, rp		C offset rp
	blr		%r28, %r0		C branch into loop
	INITCY

LDEF(loop)
	ldd		0(up), %r20
	ldd		0(vp), %r31
	ADCSBC		%r20, %r31, %r20
	std		%r20, 0(rp)
LDEF(7)	ldd		8(up), %r21
	ldd		8(vp), %r19
	ADCSBC		%r21, %r19, %r21
	std		%r21, 8(rp)
LDEF(6)	ldd		16(up), %r20
	ldd		16(vp), %r31
	ADCSBC		%r20, %r31, %r20
	std		%r20, 16(rp)
LDEF(5)	ldd		24(up), %r21
	ldd		24(vp), %r19
	ADCSBC		%r21, %r19, %r21
	std		%r21, 24(rp)
LDEF(4)	ldd		32(up), %r20
	ldd		32(vp), %r31
	ADCSBC		%r20, %r31, %r20
	std		%r20, 32(rp)
LDEF(3)	ldd		40(up), %r21
	ldd		40(vp), %r19
	ADCSBC		%r21, %r19, %r21
	std		%r21, 40(rp)
LDEF(2)	ldd		48(up), %r20
	ldd		48(vp), %r31
	ADCSBC		%r20, %r31, %r20
	std		%r20, 48(rp)
LDEF(1)	ldd		56(up), %r21
	ldd		56(vp), %r19
	ADCSBC		%r21, %r19, %r21
	ldo		64(up), up
	std		%r21, 56(rp)
	ldo		64(vp), vp
	addib,>		-8, n, L(loop)
	ldo		64(rp), rp

	add,dc		%r0, %r0, %r29
ifdef(`OPERATION_sub_n',`
	subi		1, %r29, %r29
')
	bve		(%r2)
ifdef(`HAVE_ABI_2_0w',
`	copy		%r29, %r28
',`	ldi		0, %r28
')
EPILOGUE()
