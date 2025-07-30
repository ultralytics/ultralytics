dnl  PowerPC-32 mpn_sec_tabselect.

dnl  Contributed to the GNU project by Torbjörn Granlund.

dnl  Copyright 2011-2013 Free Software Foundation, Inc.

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

C                   cycles/limb
C 603e:			 ?
C 604e:			 ?
C 75x (G3):		 ?
C 7400,7410 (G4):	 2.5
C 744x,745x (G4+):	 2.0
C power4/ppc970:	 2.0
C power5:		 ?

define(`rp',     `r3')
define(`tp',     `r4')
define(`n',      `r5')
define(`nents',  `r6')
define(`which',  `r7')

define(`i',      `r8')
define(`j',      `r9')
define(`stride', `r12')
define(`mask',   `r11')


ASM_START()
PROLOGUE(mpn_sec_tabselect)
	stwu	r1, -32(r1)
	addic.	j, n, -4		C outer loop induction variable
	stmw	r27, 8(r1)
	slwi	stride, n, 2

	blt	cr0, L(outer_end)
L(outer_top):
	mtctr	nents
	mr	r10, tp
	li	r28, 0
	li	r29, 0
	li	r30, 0
	li	r31, 0
	addic.	j, j, -4		C outer loop induction variable
	mr	i, which

	ALIGN(16)
L(top):	addic	i, i, -1		C set carry iff i != 0
	subfe	mask, mask, mask
	lwz	r0, 0(tp)
	lwz	r27, 4(tp)
	and	r0, r0, mask
	and	r27, r27, mask
	or	r28, r28, r0
	or	r29, r29, r27
	lwz	r0, 8(tp)
	lwz	r27, 12(tp)
	and	r0, r0, mask
	and	r27, r27, mask
	or	r30, r30, r0
	or	r31, r31, r27
	add	tp, tp, stride
	bdnz	L(top)

	stw	r28, 0(rp)
	stw	r29, 4(rp)
	stw	r30, 8(rp)
	stw	r31, 12(rp)
	addi	tp, r10, 16
	addi	rp, rp, 16
	bge	cr0, L(outer_top)
L(outer_end):

	andi.	r0, n, 2
	beq	cr0, L(b0x)
L(b1x):	mtctr	nents
	mr	r10, tp
	li	r28, 0
	li	r29, 0
	mr	i, which
	ALIGN(16)
L(tp2):	addic	i, i, -1
	subfe	mask, mask, mask
	lwz	r0, 0(tp)
	lwz	r27, 4(tp)
	and	r0, r0, mask
	and	r27, r27, mask
	or	r28, r28, r0
	or	r29, r29, r27
	add	tp, tp, stride
	bdnz	L(tp2)
	stw	r28, 0(rp)
	stw	r29, 4(rp)
	addi	tp, r10, 8
	addi	rp, rp, 8

L(b0x):	andi.	r0, n, 1
	beq	cr0, L(b00)
L(b01):	mtctr	nents
	mr	r10, tp
	li	r28, 0
	mr	i, which
	ALIGN(16)
L(tp1):	addic	i, i, -1
	subfe	mask, mask, mask
	lwz	r0, 0(tp)
	and	r0, r0, mask
	or	r28, r28, r0
	add	tp, tp, stride
	bdnz	L(tp1)
	stw	r28, 0(rp)

L(b00):	lmw	r27, 8(r1)
	addi	r1, r1, 32
	blr
EPILOGUE()
