dnl  PowerPC-64 mpn_mod_34lsub1 -- modulo 2^48-1.

dnl  Copyright 2005 Free Software Foundation, Inc.

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
C POWER3/PPC630          1.33
C POWER4/PPC970          1.5
C POWER5                 1.32
C POWER6                 2.35
C POWER7                 1

C INPUT PARAMETERS
define(`up',`r3')
define(`n',`r4')

ASM_START()
PROLOGUE(mpn_mod_34lsub1)
	li	r8, 0
	li	r9, 0
	li	r10, 0
	li	r11, 0

	cmpdi	cr6, n, 3
	blt	cr6, L(lt3)

	li	r0, -0x5556		C 0xFFFFFFFFFFFFAAAA
	rldimi	r0, r0, 16, 32		C 0xFFFFFFFFAAAAAAAA
	rldimi	r0, r0, 32, 63		C 0xAAAAAAAAAAAAAAAB
	mulhdu	r0, r0, n
	srdi	r0, r0, 1		C r0 = [n / 3]
	mtctr	r0

	ld	r5, 0(up)
	ld	r6, 8(up)
	ld	r7, 16(up)
	addi	up, up, 24
	bdz	L(end)

	ALIGN(16)
L(top):	addc	r8, r8, r5
	nop
	ld	r5, 0(up)
	adde	r9, r9, r6
	ld	r6, 8(up)
	adde	r10, r10, r7
	ld	r7, 16(up)
	addi	up, up, 48
	addze	r11, r11
	bdz	L(endx)
	addc	r8, r8, r5
	nop
	ld	r5, -24(up)
	adde	r9, r9, r6
	ld	r6, -16(up)
	adde	r10, r10, r7
	ld	r7, -8(up)
	addze	r11, r11
	bdnz	L(top)

	addi	up, up, 24
L(endx):
	addi	up, up, -24

L(end):	addc	r8, r8, r5
	adde	r9, r9, r6
	adde	r10, r10, r7
	addze	r11, r11

	sldi	r5, r0, 1
	add	r5, r5, r0		C r11 = n / 3 * 3
	sub	n, n, r5		C n = n mod 3
L(lt3):	cmpdi	cr6, n, 1
	blt	cr6, L(2)

	ld	r5, 0(up)
	addc	r8, r8, r5
	li	r6, 0
	beq	cr6, L(1)

	ld	r6, 8(up)
L(1):	adde	r9, r9, r6
	addze	r10, r10
	addze	r11, r11

L(2):	rldicl	r0, r8, 0, 16		C r0 = r8 mod 2^48
	srdi	r3, r8, 48		C r3 = r8 div 2^48
	rldic	r4, r9, 16, 16		C r4 = (r9 mod 2^32) << 16
	srdi	r5, r9, 32		C r5 = r9 div 2^32
	rldic	r6, r10, 32, 16		C r6 = (r10 mod 2^16) << 32
	srdi	r7, r10, 16		C r7 = r10 div 2^16

	add	r0, r0, r3
	add	r4, r4, r5
	add	r6, r6, r7

	add	r0, r0, r4
	add	r6, r6, r11

	add	r3, r0, r6
	blr
EPILOGUE()

C |__r10__|__r9___|__r8___|
C |-----|-----|-----|-----|
