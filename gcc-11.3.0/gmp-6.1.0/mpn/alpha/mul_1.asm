dnl  Alpha mpn_mul_1 -- Multiply a limb vector with a limb and store
dnl  the result in a second limb vector.

dnl  Copyright 1992, 1994, 1995, 2000, 2002 Free Software Foundation, Inc.

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
C EV4:     42
C EV5:     18
C EV6:      7

C  INPUT PARAMETERS
C  rp	r16
C  up	r17
C  n	r18
C  vl	r19
C  cl	r20


ASM_START()
PROLOGUE(mpn_mul_1c)
	ldq	r2,0(r17)	C r2 = s1_limb
	lda	r18,-1(r18)	C size--
	mulq	r2,r19,r3	C r3 = prod_low
	umulh	r2,r19,r4	C r4 = prod_high
	beq	r18,$Le1c	C jump if size was == 1
	ldq	r2,8(r17)	C r2 = s1_limb
	lda	r18,-1(r18)	C size--
	addq	r3,r20,r3	C r3 = cy_limb + cl
	stq	r3,0(r16)
	cmpult	r3,r20,r0	C r0 = carry from (cy_limb + cl)
	bne	r18,$Loop	C jump if size was == 2
	br	r31,$Le2
$Le1c:	addq	r3,r20,r3	C r3 = cy_limb + cl
	cmpult	r3,r20,r0	C r0 = carry from (cy_limb + cl)
$Le1:	stq	r3,0(r16)
	addq	r4,r0,r0
	ret	r31,(r26),1
EPILOGUE(mpn_mul_1c)

PROLOGUE(mpn_mul_1)
	ldq	r2,0(r17)	C r2 = s1_limb
	lda	r18,-1(r18)	C size--
	mulq	r2,r19,r3	C r3 = prod_low
	bic	r31,r31,r0	C clear cy_limb
	umulh	r2,r19,r4	C r4 = prod_high
	beq	r18,$Le1	C jump if size was == 1
	ldq	r2,8(r17)	C r2 = s1_limb
	lda	r18,-1(r18)	C size--
	stq	r3,0(r16)
	beq	r18,$Le2	C jump if size was == 2

	ALIGN(8)
$Loop:	mulq	r2,r19,r3	C r3 = prod_low
	addq	r4,r0,r0	C cy_limb = cy_limb + 'cy'
	lda	r18,-1(r18)	C size--
	umulh	r2,r19,r4	C r4 = prod_high
	ldq	r2,16(r17)	C r2 = s1_limb
	lda	r17,8(r17)	C s1_ptr++
	addq	r3,r0,r3	C r3 = cy_limb + prod_low
	stq	r3,8(r16)
	cmpult	r3,r0,r0	C r0 = carry from (cy_limb + prod_low)
	lda	r16,8(r16)	C res_ptr++
	bne	r18,$Loop

$Le2:	mulq	r2,r19,r3	C r3 = prod_low
	addq	r4,r0,r0	C cy_limb = cy_limb + 'cy'
	umulh	r2,r19,r4	C r4 = prod_high
	addq	r3,r0,r3	C r3 = cy_limb + prod_low
	cmpult	r3,r0,r0	C r0 = carry from (cy_limb + prod_low)
	stq	r3,8(r16)
	addq	r4,r0,r0	C cy_limb = prod_high + cy
	ret	r31,(r26),1
EPILOGUE(mpn_mul_1)
ASM_END()
