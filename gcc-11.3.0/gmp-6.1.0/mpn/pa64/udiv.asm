dnl  HP-PA 2.0 64-bit mpn_udiv_qrnnd_r.

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

include(`../config.m4')

C This runs at about 280 cycles on both PA8000 and PA8500, corresponding to a
C bit more than 4 cycles/bit.

C INPUT PARAMETERS
define(`n1',`%r26')
define(`n0',`%r25')
define(`d',`%r24')
define(`remptr',`%r23')

define(`q',`%r28')
define(`dn',`%r29')

define(`old_divstep',
       `add,dc		n0,n0,n0
	add,dc		n1,n1,n1
	sub,*<<		n1,d,%r22
	copy		%r22,n1')

define(`divstep',
       `add		n0,n0,n0
	add,dc		n1,n1,n1
	sub		n1,d,%r1
	add,dc		q,q,q
	cmpclr,*<<	n1,d,%r0
	copy		%r1,n1
')

ifdef(`HAVE_ABI_2_0w',
`	.level	2.0w
',`	.level	2.0
')
PROLOGUE(mpn_udiv_qrnnd_r)
ifdef(`HAVE_ABI_2_0n',
`	depd		%r25,31,32,%r26
	depd		%r23,31,32,%r24
	copy		%r24,%r25
	ldd		-56(%r30),%r24
	ldw		-60(%r30),%r23
')
	ldi		0,q
	cmpib,*>=	0,d,L(large_divisor)
	ldi		8,%r31		C setup loop counter

	sub		%r0,d,dn
LDEF(Loop)
	divstep divstep divstep divstep divstep divstep divstep divstep
	addib,<>	-1,%r31,L(Loop)
	nop

ifdef(`HAVE_ABI_2_0n',
`	copy		%r28,%r29
	extrd,u		%r28,31,32,%r28
')
	bve		(%r2)
	std		n1,0(remptr)	C store remainder

LDEF(large_divisor)
	extrd,u		n0,63,1,%r19	C save lsb of dividend
	shrpd		n1,n0,1,n0	C n0 = lo(n1n0 >> 1)
	shrpd		%r0,n1,1,n1	C n1 = hi(n1n0 >> 1)
	extrd,u		d,63,1,%r20	C save lsb of divisor
	shrpd		%r0,d,1,d	C d = floor(orig_d / 2)
	add,l		%r20,d,d	C d = ceil(orig_d / 2)

	sub		%r0,d,dn
LDEF(Loop2)
	divstep divstep divstep divstep divstep divstep divstep divstep
	addib,<>	-1,%r31,L(Loop2)
	nop

	cmpib,*=	0,%r20,L(even_divisor)
	shladd		n1,1,%r19,n1	C shift in omitted dividend lsb

	add		d,d,d		C restore orig...
	sub		d,%r20,d	C ...d value
	sub		%r0,d,dn	C r21 = -d

	add,*nuv	n1,q,n1		C fix remainder for omitted divisor lsb
	add,l		n1,dn,n1	C adjust remainder if rem. fix carried
	add,dc		%r0,q,q		C adjust quotient accordingly

	sub,*<<		n1,d,%r0	C remainder >= divisor?
	add,l		n1,dn,n1	C adjust remainder
	add,dc		%r0,q,q		C adjust quotient

LDEF(even_divisor)
ifdef(`HAVE_ABI_2_0n',
`	copy		%r28,%r29
	extrd,u		%r28,31,32,%r28
')
	bve		(%r2)
	std		n1,0(remptr)	C store remainder
EPILOGUE(mpn_udiv_qrnnd_r)
