dnl  Alpha ev6 nails mpn_add_n and mpn_sub_n.

dnl  Copyright 2002, 2006 Free Software Foundation, Inc.

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


dnl  Runs at 2.5 cycles/limb.  It would be possible to reach 2.0 cycles/limb
dnl  with 8-way unrolling.

include(`../config.m4')

dnl  INPUT PARAMETERS
define(`rp',`r16')
define(`up',`r17')
define(`vp',`r18')
define(`n',`r19')

define(`rl0',`r0')
define(`rl1',`r1')
define(`rl2',`r2')
define(`rl3',`r3')

define(`ul0',`r4')
define(`ul1',`r5')
define(`ul2',`r6')
define(`ul3',`r7')

define(`vl0',`r22')
define(`vl1',`r23')
define(`vl2',`r24')
define(`vl3',`r25')

define(`numb_mask',`r21')

define(`NAIL_BITS',`GMP_NAIL_BITS')
define(`CYSH',`GMP_NUMB_BITS')

dnl  This declaration is munged by configure
NAILS_SUPPORT(1-63)

ifdef(`OPERATION_add_n', `
	define(`OP',        addq)
	define(`CYSH',`GMP_NUMB_BITS')
	define(`func',  mpn_add_n)')
ifdef(`OPERATION_sub_n', `
	define(`OP',        subq)
	define(`CYSH',63)
	define(`func',  mpn_sub_n)')

MULFUNC_PROLOGUE(mpn_add_n mpn_sub_n)

ASM_START()
PROLOGUE(func)
	lda	numb_mask, -1(r31)
	srl	numb_mask, NAIL_BITS, numb_mask
	bis	r31,	r31,	r20

	and	n,	3,	r25
	lda	n,	-4(n)
	beq	r25,	L(ge4)

L(lp0):	ldq	ul0,	0(up)
	lda	up,	8(up)
	ldq	vl0,	0(vp)
	lda	vp,	8(vp)
	lda	rp,	8(rp)
	lda	r25,	-1(r25)
	OP	ul0,	vl0,	rl0
	OP	rl0,	r20,	rl0
	and	rl0, numb_mask,	r28
	stq	r28,	-8(rp)
	srl	rl0,	CYSH,	r20
	bne	r25,	L(lp0)

	blt	n,	L(ret)

L(ge4):	ldq	ul0,	0(up)
	ldq	vl0,	0(vp)
	ldq	ul1,	8(up)
	ldq	vl1,	8(vp)
	ldq	ul2,	16(up)
	ldq	vl2,	16(vp)
	ldq	ul3,	24(up)
	ldq	vl3,	24(vp)
	lda	up,	32(up)
	lda	vp,	32(vp)
	lda	n,	-4(n)
	bge	n,	L(ge8)

	OP	ul0,	vl0,	rl0	C		main-add 0
	OP	rl0,	r20,	rl0	C		cy-add 0
	OP	ul1,	vl1,	rl1	C		main-add 1
	srl	rl0,	CYSH,	r20	C		gen cy 0
	OP	rl1,	r20,	rl1	C		cy-add 1
	and	rl0,numb_mask,	r27
	br	r31,	L(cj0)

L(ge8):	OP	ul0,	vl0,	rl0	C		main-add 0
	ldq	ul0,	0(up)
	ldq	vl0,	0(vp)
	OP	rl0,	r20,	rl0	C		cy-add 0
	OP	ul1,	vl1,	rl1	C		main-add 1
	srl	rl0,	CYSH,	r20	C		gen cy 0
	ldq	ul1,	8(up)
	ldq	vl1,	8(vp)
	OP	rl1,	r20,	rl1	C		cy-add 1
	and	rl0,numb_mask,	r27
	OP	ul2,	vl2,	rl2	C		main-add 2
	srl	rl1,	CYSH,	r20	C		gen cy 1
	ldq	ul2,	16(up)
	ldq	vl2,	16(vp)
	OP	rl2,	r20,	rl2	C		cy-add 2
	and	rl1,numb_mask,	r28
	stq	r27,	0(rp)
	OP	ul3,	vl3,	rl3	C		main-add 3
	srl	rl2,	CYSH,	r20	C		gen cy 2
	ldq	ul3,	24(up)
	ldq	vl3,	24(vp)
	OP	rl3,	r20,	rl3	C		cy-add 3
	and	rl2,numb_mask,	r27
	stq	r28,	8(rp)
	lda	rp,	32(rp)
	lda	up,	32(up)
	lda	vp,	32(vp)
	lda	n,	-4(n)
	blt	n,	L(end)

	ALIGN(32)
L(top):	OP	ul0,	vl0,	rl0	C		main-add 0
	srl	rl3,	CYSH,	r20	C		gen cy 3
	ldq	ul0,	0(up)
	ldq	vl0,	0(vp)

	OP	rl0,	r20,	rl0	C		cy-add 0
	and	rl3,numb_mask,	r28
	stq	r27,	-16(rp)
	bis	r31,	r31,	r31

	OP	ul1,	vl1,	rl1	C		main-add 1
	srl	rl0,	CYSH,	r20	C		gen cy 0
	ldq	ul1,	8(up)
	ldq	vl1,	8(vp)

	OP	rl1,	r20,	rl1	C		cy-add 1
	and	rl0,numb_mask,	r27
	stq	r28,	-8(rp)
	bis	r31,	r31,	r31

	OP	ul2,	vl2,	rl2	C		main-add 2
	srl	rl1,	CYSH,	r20	C		gen cy 1
	ldq	ul2,	16(up)
	ldq	vl2,	16(vp)

	OP	rl2,	r20,	rl2	C		cy-add 2
	and	rl1,numb_mask,	r28
	stq	r27,	0(rp)
	bis	r31,	r31,	r31

	OP	ul3,	vl3,	rl3	C		main-add 3
	srl	rl2,	CYSH,	r20	C		gen cy 2
	ldq	ul3,	24(up)
	ldq	vl3,	24(vp)

	OP	rl3,	r20,	rl3	C		cy-add 3
	and	rl2,numb_mask,	r27
	stq	r28,	8(rp)
	bis	r31,	r31,	r31

	bis	r31,	r31,	r31
	lda	n,	-4(n)
	lda	up,	32(up)
	lda	vp,	32(vp)

	bis	r31,	r31,	r31
	bis	r31,	r31,	r31
	lda	rp,	32(rp)
	bge	n,	L(top)

L(end):	OP	ul0,	vl0,	rl0	C		main-add 0
	srl	rl3,	CYSH,	r20	C		gen cy 3
	OP	rl0,	r20,	rl0	C		cy-add 0
	and	rl3,numb_mask,	r28
	stq	r27,	-16(rp)
	OP	ul1,	vl1,	rl1	C		main-add 1
	srl	rl0,	CYSH,	r20	C		gen cy 0
	OP	rl1,	r20,	rl1	C		cy-add 1
	and	rl0,numb_mask,	r27
	stq	r28,	-8(rp)
L(cj0):	OP	ul2,	vl2,	rl2	C		main-add 2
	srl	rl1,	CYSH,	r20	C		gen cy 1
	OP	rl2,	r20,	rl2	C		cy-add 2
	and	rl1,numb_mask,	r28
	stq	r27,	0(rp)
	OP	ul3,	vl3,	rl3	C		main-add 3
	srl	rl2,	CYSH,	r20	C		gen cy 2
	OP	rl3,	r20,	rl3	C		cy-add 3
	and	rl2,numb_mask,	r27
	stq	r28,	8(rp)

	srl	rl3,	CYSH,	r20	C		gen cy 3
	and	rl3,numb_mask,	r28
	stq	r27,	16(rp)
	stq	r28,	24(rp)

L(ret):	and	r20,	1,	r0
	ret	r31,	(r26),	1
EPILOGUE()
ASM_END()
