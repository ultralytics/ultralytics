dnl  PowerPC-64 mpn_copyd.

dnl  Copyright 2004, 2005, 2013 Free Software Foundation, Inc.

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

C                  cycles/limb
C POWER3/PPC630          ?
C POWER4/PPC970          ?
C POWER5                 ?
C POWER6                 1.25
C POWER7                 1.09

C INPUT PARAMETERS
define(`rp',	`r3')
define(`up',	`r4')
define(`n',	`r5')

ASM_START()
PROLOGUE(mpn_copyd)

ifdef(`HAVE_ABI_mode32',
`	rldicl	n, n, 0,32')

	sldi	r0, n, 3
	add	up, up, r0		C point at u[] end
	add	rp, rp, r0		C point at r[] end

	cmpdi	cr0, n, 4
	blt	L(sml)

	addi	r10, n, 4
	srdi	r10, r10, 3
	mtctr	r10

	andi.	r0, n, 1
	rlwinm	r11, n, 0,30,30
	rlwinm	r12, n, 0,29,29
	cmpdi	cr6, r11, 0
	cmpdi	cr7, r12, 0

	beq	cr0, L(xx0)
L(xx1):	ld	r6, -8(up)
	addi	up, up, -8
	std	r6, -8(rp)
	addi	rp, rp, -8

L(xx0):	bne	cr6, L(x10)
L(x00):	ld	r6, -8(up)
	ld	r7, -16(up)
	bne	cr7, L(100)
L(000):	addi	rp, rp, 32
	b	L(lo0)
L(100):	addi	up, up, 32
	b	L(lo4)
L(x10):	ld	r8, -8(up)
	ld	r9, -16(up)
	bne	cr7, L(110)
L(010):	addi	up, up, -16
	addi	rp, rp, 16
	b	L(lo2)
L(110):	addi	up, up, 16
	addi	rp, rp, 48
	b	L(lo6)

L(sml):	cmpdi	cr0, n, 0
	beqlr-	cr0
	mtctr	n
L(t):	ld	r6, -8(up)
	addi	up, up, -8
	std	r6, -8(rp)
	addi	rp, rp, -8
	bdnz	L(t)
	blr

	ALIGN(32)
L(top):	std	r6, -8(rp)
	std	r7, -16(rp)
L(lo2):	ld	r6, -8(up)
	ld	r7, -16(up)
	std	r8, -24(rp)
	std	r9, -32(rp)
L(lo0):	ld	r8, -24(up)
	ld	r9, -32(up)
	std	r6, -40(rp)
	std	r7, -48(rp)
L(lo6):	ld	r6, -40(up)
	ld	r7, -48(up)
	std	r8, -56(rp)
	std	r9, -64(rp)
	addi	rp, rp, -64
L(lo4):	ld	r8, -56(up)
	ld	r9, -64(up)
	addi	up, up, -64
	bdnz	L(top)

L(end):	std	r6, -8(rp)
	std	r7, -16(rp)
	std	r8, -24(rp)
	std	r9, -32(rp)
	blr
EPILOGUE()
