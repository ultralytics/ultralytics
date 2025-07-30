dnl  PA64 mpn_addlsh1_n/mpn_sublsh1_n -- rp[] = up[] +- (vp[] << 1).

dnl  Copyright 2003 Free Software Foundation, Inc.

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

C		    cycles/limb
C 8000,8200:		2
C 8500,8600,8700:	1.75

C TODO
C  * Write special feed-in code for each (n mod 8). (See the ia64 code.)
C  * Try to make this run at closer to 1.5 c/l.
C  * Set up register aliases (define(`u0',`%r19')).
C  * Explicitly align loop.

dnl INPUT PARAMETERS
define(`rp',`%r26')
define(`up',`%r25')
define(`vp',`%r24')
define(`n',`%r23')

ifdef(`OPERATION_addlsh1_n',`
  define(ADCSBC,	`add,dc')
  define(INITC,		`ldi	0,')
  define(func, mpn_addlsh1_n)
')
ifdef(`OPERATION_sublsh1_n',`
  define(ADCSBC,	`sub,db')
  define(INITC,		`ldi	1,')
  define(func, mpn_sublsh1_n)
')

MULFUNC_PROLOGUE(mpn_addlsh1_n mpn_sublsh1_n)

ifdef(`HAVE_ABI_2_0w',`
  define(LEVEL,		`.level 2.0w')
  define(RETREG,	`%r28')
  define(CLRRET1,	`dnl')
')
ifdef(`HAVE_ABI_2_0n',`
  define(LEVEL,		`.level 2.0')
  define(RETREG,	`%r29')
  define(CLRRET1,	`ldi	0, %r28')
')

	LEVEL
PROLOGUE(func)
	std,ma		%r3, 0x100(%r30)	C save reg

	INITC		%r1			C init saved cy

C Primitive code for the first (n mod 8) limbs:
	extrd,u		n, 63, 3, %r22		C count for loop0
	comib,=		0, %r22, L(unrolled)	C skip loop0?
	copy		%r0, %r28
LDEF(loop0)
	ldd	0(vp), %r21
	ldo		8(vp), vp
	ldd	0(up), %r19
	ldo		8(up), up
	shrpd	%r21, %r28, 63, %r31
	addi		-1, %r1, %r0		C restore cy
	ADCSBC	%r19, %r31, %r29
	std	%r29, 0(rp)
	add,dc		%r0, %r0, %r1		C save cy
	copy	%r21, %r28
	addib,>		-1, %r22, L(loop0)
	ldo		8(rp), rp

	addib,>=	-8, n, L(unrolled)
	addi		-1, %r1, %r0		C restore cy

	shrpd	%r0, %r28, 63, %r28
	ADCSBC	%r0, %r28, RETREG
ifdef(`OPERATION_sublsh1_n',
`	sub	%r0, RETREG, RETREG')
	CLRRET1

	bve		(%r2)
	ldd,mb		-0x100(%r30), %r3


LDEF(unrolled)
	std		%r4, -0xf8(%r30)	C save reg
	ldd	0(vp), %r4
	std		%r5, -0xf0(%r30)	C save reg
	ldd	8(vp), %r5
	std		%r6, -0xe8(%r30)	C save reg
	ldd	16(vp), %r6
	std		%r7, -0xe0(%r30)	C save reg

	ldd	24(vp), %r7
	shrpd	%r4, %r28, 63, %r31
	std		%r8, -0xd8(%r30)	C save reg
	ldd	32(vp), %r8
	shrpd	%r5, %r4, 63, %r4
	std		%r9, -0xd0(%r30)	C save reg
	ldd	40(vp), %r9
	shrpd	%r6, %r5, 63, %r5
	ldd	48(vp), %r3
	shrpd	%r7, %r6, 63, %r6
	ldd	56(vp), %r28
	shrpd	%r8, %r7, 63, %r7
	ldd	0(up), %r19
	shrpd	%r9, %r8, 63, %r8
	ldd	8(up), %r20
	shrpd	%r3, %r9, 63, %r9
	ldd	16(up), %r21
	shrpd	%r28, %r3, 63, %r3
	ldd	24(up), %r22

	nop					C alignment FIXME
	addib,<=	-8, n, L(end)
	addi		-1, %r1, %r0		C restore cy
LDEF(loop)
	ADCSBC	%r19, %r31, %r29
	ldd	32(up), %r19
	std	%r29, 0(rp)
	ADCSBC	%r20, %r4, %r29
	ldd	40(up), %r20
	std	%r29, 8(rp)
	ADCSBC	%r21, %r5, %r29
	ldd	48(up), %r21
	std	%r29, 16(rp)
	ADCSBC	%r22, %r6, %r29
	ldd	56(up), %r22
	std	%r29, 24(rp)
	ADCSBC	%r19, %r7, %r29
	ldd	64(vp), %r4
	std	%r29, 32(rp)
	ADCSBC	%r20, %r8, %r29
	ldd	72(vp), %r5
	std	%r29, 40(rp)
	ADCSBC	%r21, %r9, %r29
	ldd	80(vp), %r6
	std	%r29, 48(rp)
	ADCSBC	%r22, %r3, %r29
	std	%r29, 56(rp)

	add,dc		%r0, %r0, %r1		C save cy

	ldd	88(vp), %r7
	shrpd	%r4, %r28, 63, %r31
	ldd	96(vp), %r8
	shrpd	%r5, %r4, 63, %r4
	ldd	104(vp), %r9
	shrpd	%r6, %r5, 63, %r5
	ldd	112(vp), %r3
	shrpd	%r7, %r6, 63, %r6
	ldd	120(vp), %r28
	shrpd	%r8, %r7, 63, %r7
	ldd	64(up), %r19
	shrpd	%r9, %r8, 63, %r8
	ldd	72(up), %r20
	shrpd	%r3, %r9, 63, %r9
	ldd	80(up), %r21
	shrpd	%r28, %r3, 63, %r3
	ldd	88(up), %r22

	ldo		64(vp), vp
	ldo		64(rp), rp
	ldo		64(up), up
	addib,>		-8, n, L(loop)
	addi		-1, %r1, %r0		C restore cy
LDEF(end)
	ADCSBC	%r19, %r31, %r29
	ldd	32(up), %r19
	std	%r29, 0(rp)
	ADCSBC	%r20, %r4, %r29
	ldd	40(up), %r20
	std	%r29, 8(rp)
	ADCSBC	%r21, %r5, %r29
	ldd	48(up), %r21
	std	%r29, 16(rp)
	ADCSBC	%r22, %r6, %r29
	ldd	56(up), %r22
	std	%r29, 24(rp)
	ADCSBC	%r19, %r7, %r29
	ldd		-0xf8(%r30), %r4	C restore reg
	std	%r29, 32(rp)
	ADCSBC	%r20, %r8, %r29
	ldd		-0xf0(%r30), %r5	C restore reg
	std	%r29, 40(rp)
	ADCSBC	%r21, %r9, %r29
	ldd		-0xe8(%r30), %r6	C restore reg
	std	%r29, 48(rp)
	ADCSBC	%r22, %r3, %r29
	ldd		-0xe0(%r30), %r7	C restore reg
	std	%r29, 56(rp)

	shrpd	%r0, %r28, 63, %r28
	ldd		-0xd8(%r30), %r8	C restore reg
	ADCSBC	%r0, %r28, RETREG
ifdef(`OPERATION_sublsh1_n',
`	sub	%r0, RETREG, RETREG')
	CLRRET1

	ldd		-0xd0(%r30), %r9	C restore reg
	bve		(%r2)
	ldd,mb		-0x100(%r30), %r3	C restore reg
EPILOGUE()
