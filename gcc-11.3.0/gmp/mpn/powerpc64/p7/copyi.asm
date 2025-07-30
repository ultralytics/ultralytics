dnl  PowerPC-64 mpn_copyi.

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

C TODO
C  * Try rolling the two loop leading std to the end, allowing the code to
C    handle also n = 2.
C  * Consider using 4 pointers, schedule ptr update early wrt use.

ASM_START()
PROLOGUE(mpn_copyi)

ifdef(`HAVE_ABI_mode32',
`	rldicl	n, n, 0,32')

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
L(xx1):	ld	r6, 0(up)
	addi	up, up, 8
	std	r6, 0(rp)
	addi	rp, rp, 8

L(xx0):	bne	cr6, L(x10)
L(x00):	ld	r6, 0(up)
	ld	r7, 8(up)
	bne	cr7, L(100)
L(000):	addi	rp, rp, -32
	b	L(lo0)
L(100):	addi	up, up, -32
	b	L(lo4)
L(x10):	ld	r8, 0(up)
	ld	r9, 8(up)
	bne	cr7, L(110)
L(010):	addi	up, up, 16
	addi	rp, rp, -16
	b	L(lo2)
L(110):	addi	up, up, -16
	addi	rp, rp, -48
	b	L(lo6)

L(sml):	cmpdi	cr0, n, 0
	beqlr-	cr0
	mtctr	n
L(t):	ld	r6, 0(up)
	addi	up, up, 8
	std	r6, 0(rp)
	addi	rp, rp, 8
	bdnz	L(t)
	blr

	ALIGN(32)
L(top):	std	r6, 0(rp)
	std	r7, 8(rp)
L(lo2):	ld	r6, 0(up)
	ld	r7, 8(up)
	std	r8, 16(rp)
	std	r9, 24(rp)
L(lo0):	ld	r8, 16(up)
	ld	r9, 24(up)
	std	r6, 32(rp)
	std	r7, 40(rp)
L(lo6):	ld	r6, 32(up)
	ld	r7, 40(up)
	std	r8, 48(rp)
	std	r9, 56(rp)
	addi	rp, rp, 64
L(lo4):	ld	r8, 48(up)
	ld	r9, 56(up)
	addi	up, up, 64
	bdnz	L(top)

L(end):	std	r6, 0(rp)
	std	r7, 8(rp)
	std	r8, 16(rp)
	std	r9, 24(rp)
	blr
EPILOGUE()
