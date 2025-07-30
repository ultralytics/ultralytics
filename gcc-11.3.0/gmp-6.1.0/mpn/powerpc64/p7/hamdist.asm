dnl  PowerPC-64 mpn_hamdist.

dnl  Copyright 2012 Free Software Foundation, Inc.

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
C POWER3/PPC630          -
C POWER4/PPC970          -
C POWER5                 -
C POWER6                 -
C POWER7                 2.87

define(`up', r3)
define(`vp', r4)
define(`n',  r5)

ASM_START()
PROLOGUE(mpn_hamdist)
	std	r30, -16(r1)
	std	r31, -8(r1)

	addi	r0, n, 1
ifdef(`HAVE_ABI_mode32',
`	rldicl	r0, r0, 63,33',	C ...branch count
`	srdi	r0, r0, 1')	C ...for ctr
	mtctr	r0

	andi.	r0, n, 1

	li	r0, 0
	li	r12, 0

	beq	L(evn)

L(odd):	ld	r6, 0(up)
	addi	up, up, 8
	ld	r8, 0(vp)
	addi	vp, vp, 8
	xor	r10, r6, r8
	popcntd	r0, r10
	bdz	L(e1)

L(evn):	ld	r6, 0(up)
	ld	r8, 0(vp)
	ld	r7, 8(up)
	ld	r9, 8(vp)
	xor	r10, r6, r8
	addi	up, up, 16
	addi	vp, vp, 16
	li	r30, 0
	li	r31, 0
	bdz	L(end)

	nop
	nop
C	ALIGN(16)
L(top):	add	r0, r0, r30
	ld	r6, 0(up)
	ld	r8, 0(vp)
	xor	r11, r7, r9
	popcntd	r30, r10
	add	r12, r12, r31
	ld	r7, 8(up)
	ld	r9, 8(vp)
	xor	r10, r6, r8
	popcntd	r31, r11
	addi	up, up, 16
	addi	vp, vp, 16
	bdnz	L(top)

L(end):	add	r0, r0, r30
	xor	r11, r7, r9
	popcntd	r30, r10
	add	r12, r12, r31
	popcntd	r31, r11

	add	r0, r0, r30
	add	r12, r12, r31
L(e1):	add	r3, r0, r12
	ld	r30, -16(r1)
	ld	r31, -8(r1)
	blr
EPILOGUE()
