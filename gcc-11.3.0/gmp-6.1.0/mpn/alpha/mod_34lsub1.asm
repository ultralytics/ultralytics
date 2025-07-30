dnl Alpha mpn_mod_34lsub1.

dnl  Copyright 2002 Free Software Foundation, Inc.

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

C      cycles/limb
C EV4:     4 (?)
C EV5:     2.67
C EV6:     1.67


dnl  INPUT PARAMETERS
dnl  up		r16
dnl  n		r17

define(`l0',`r18')
define(`l1',`r19')
define(`l2',`r20')
define(`a0',`r21')
define(`a1',`r22')
define(`a2',`r23')
define(`c0',`r24')
define(`c1',`r5')
define(`c2',`r6')

ASM_START()
PROLOGUE(mpn_mod_34lsub1)
	bis	r31, r31, c0
	bis	r31, r31, c1
	bis	r31, r31, c2

	lda	r17, -3(r17)
	bge	r17, $L_3_or_more
	bis	r31, r31, a0
	bis	r31, r31, a1
	bis	r31, r31, a2
	br	r31, $L_012

$L_3_or_more:
	ldq	a0, 0(r16)
	ldq	a1, 8(r16)
	ldq	a2, 16(r16)
	lda	r16, 24(r16)
	lda	r17, -3(r17)
	blt	r17, $L_012

$L_6_or_more:
	ldq	l0, 0(r16)
	ldq	l1, 8(r16)
	ldq	l2, 16(r16)
	addq	l0, a0, a0

	lda	r16, 24(r16)
	lda	r17, -3(r17)
	blt	r17, $L_end

	ALIGN(16)
C Main loop
$L_9_or_more:
$Loop:	cmpult	a0, l0, r0
	ldq	l0, 0(r16)
	addq	r0, c0, c0
	addq	l1, a1, a1
	cmpult	a1, l1, r0
	ldq	l1, 8(r16)
	addq	r0, c1, c1
	addq	l2, a2, a2
	cmpult	a2, l2, r0
	ldq	l2, 16(r16)
	addq	r0, c2, c2
	addq	l0, a0, a0
	lda	r16, 24(r16)
	lda	r17, -3(r17)
	bge	r17, $Loop

$L_end:	cmpult	a0, l0, r0
	addq	r0, c0, c0
	addq	l1, a1, a1
	cmpult	a1, l1, r0
	addq	r0, c1, c1
	addq	l2, a2, a2
	cmpult	a2, l2, r0
	addq	r0, c2, c2

C Handle the last (n mod 3) limbs
$L_012:	lda	r17, 2(r17)
	blt	r17, $L_0
	ldq	l0, 0(r16)
	addq	l0, a0, a0
	cmpult	a0, l0, r0
	addq	r0, c0, c0
	beq	r17, $L_0
	ldq	l1, 8(r16)
	addq	l1, a1, a1
	cmpult	a1, l1, r0
	addq	r0, c1, c1

C Align and sum our 3 main accumulators and 3 carry accumulators
$L_0:	srl	a0, 48, r2
	srl	a1, 32, r4
ifdef(`HAVE_LIMB_LITTLE_ENDIAN',
`	insll	a1, 2, r1',		C (a1 & 0xffffffff) << 16
`	zapnot	a1, 15, r25
	sll	r25, 16, r1')
	zapnot	a0, 63, r0		C a0 & 0xffffffffffff
	srl	a2, 16, a1
ifdef(`HAVE_LIMB_LITTLE_ENDIAN',
`	inswl	a2, 4, r3',		C (a2 & 0xffff) << 32
`	zapnot	a2, 3, r25
	sll	r25, 32, r3')
	addq	r1, r4, r1
	addq	r0, r2, r0
	srl	c0, 32, a2
ifdef(`HAVE_LIMB_LITTLE_ENDIAN',
`	insll	c0, 2, r4',		C (c0 & 0xffffffff) << 16
`	zapnot	c0, 15, r25
	sll	r25, 16, r4')
	addq	r0, r1, r0
	addq	r3, a1, r3
	addq	r0, r3, r0
	srl	c1, 16, c0
ifdef(`HAVE_LIMB_LITTLE_ENDIAN',
`	inswl	c1, 4, r2',		C (c1 & 0xffff) << 32
`	zapnot	c1, 3, r25
	sll	r25, 32, r2')
	addq	r4, a2, r4
C	srl	c2, 48, r3		C This will be 0 in practise
	zapnot	c2, 63, r1		C r1 = c2 & 0xffffffffffff
	addq	r0, r4, r0
	addq	r2, c0, r2
	addq	r0, r2, r0
C	addq	r1, r3, r1
	addq	r0, r1, r0

	ret	r31, (r26), 1
EPILOGUE(mpn_mod_34lsub1)
ASM_END()
