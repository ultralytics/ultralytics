dnl  PowerPC-64 mpn_rsh1add_n, mpn_rsh1sub_n

dnl  Copyright 2003, 2005, 2010, 2013 Free Software Foundation, Inc.

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

C		   cycles/limb
C POWER3/PPC630		 ?
C POWER4/PPC970		 2.9
C POWER5		 ?
C POWER6		 3.5
C POWER7		 2.25

define(`rp', `r3')
define(`up', `r4')
define(`vp', `r5')
define(`n',  `r6')

ifdef(`OPERATION_rsh1add_n', `
  define(`ADDSUBC',	`addc')
  define(`ADDSUBE',	`adde')
  define(INITCY,	`addic	$1, r1, 0')
  define(`func',	mpn_rsh1add_n)')
ifdef(`OPERATION_rsh1sub_n', `
  define(`ADDSUBC',	`subfc')
  define(`ADDSUBE',	`subfe')
  define(INITCY,	`addic	$1, r1, -1')
  define(`func',	mpn_rsh1sub_n)')

define(`s0', `r9')
define(`s1', `r7')
define(`x0', `r0')
define(`x1', `r12')
define(`u0', `r8')
define(`v0', `r10')

MULFUNC_PROLOGUE(mpn_rsh1add_n mpn_rsh1sub_n)

ASM_START()
PROLOGUE(func)
	ld	u0, 0(up)
	ld	v0, 0(vp)

	cmpdi	cr6, n, 2

	addi	r0, n, 1
	srdi	r0, r0, 2
	mtctr	r0			C copy size to count register

	andi.	r0, n, 1
	bne	cr0, L(bx1)

L(bx0):	ADDSUBC	x1, v0, u0
	ld	u0, 8(up)
	ld	v0, 8(vp)
	ADDSUBE	x0, v0, u0
	ble	cr6, L(n2)
	ld	u0, 16(up)
	ld	v0, 16(vp)
	srdi	s0, x1, 1
	rldicl	r11, x1, 0, 63		C return value
	ADDSUBE	x1, v0, u0
	andi.	n, n, 2
	bne	cr0, L(b10)
L(b00):	addi	rp, rp, -24
	b	L(lo0)
L(b10):	addi	up, up, 16
	addi	vp, vp, 16
	addi	rp, rp, -8
	b	L(lo2)

	ALIGN(16)
L(bx1):	ADDSUBC	x0, v0, u0
	ble	cr6, L(n1)
	ld	u0, 8(up)
	ld	v0, 8(vp)
	ADDSUBE	x1, v0, u0
	ld	u0, 16(up)
	ld	v0, 16(vp)
	srdi	s1, x0, 1
	rldicl	r11, x0, 0, 63		C return value
	ADDSUBE	x0, v0, u0
	andi.	n, n, 2
	bne	cr0, L(b11)
L(b01):	addi	up, up, 8
	addi	vp, vp, 8
	addi	rp, rp, -16
	b	L(lo1)
L(b11):	addi	up, up, 24
	addi	vp, vp, 24
	bdz	L(end)

	ALIGN(32)
L(top):	ld	u0, 0(up)
	ld	v0, 0(vp)
	srdi	s0, x1, 1
	rldimi	s1, x1, 63, 0
	std	s1, 0(rp)
	ADDSUBE	x1, v0, u0
L(lo2):	ld	u0, 8(up)
	ld	v0, 8(vp)
	srdi	s1, x0, 1
	rldimi	s0, x0, 63, 0
	std	s0, 8(rp)
	ADDSUBE	x0, v0, u0
L(lo1):	ld	u0, 16(up)
	ld	v0, 16(vp)
	srdi	s0, x1, 1
	rldimi	s1, x1, 63, 0
	std	s1, 16(rp)
	ADDSUBE	x1, v0, u0
L(lo0):	ld	u0, 24(up)
	ld	v0, 24(vp)
	srdi	s1, x0, 1
	rldimi	s0, x0, 63, 0
	std	s0, 24(rp)
	ADDSUBE	x0, v0, u0
	addi	up, up, 32
	addi	vp, vp, 32
	addi	rp, rp, 32
	bdnz	L(top)

L(end):	srdi	s0, x1, 1
	rldimi	s1, x1, 63, 0
	std	s1, 0(rp)
L(cj2):	srdi	s1, x0, 1
	rldimi	s0, x0, 63, 0
	std	s0, 8(rp)
L(cj1):	ADDSUBE	x1, x1, x1		C pseudo-depends on x1
	rldimi	s1, x1, 63, 0
	std	s1, 16(rp)
	mr	r3, r11
	blr

L(n1):	srdi	s1, x0, 1
	rldicl	r11, x0, 0, 63		C return value
	ADDSUBE	x1, x1, x1		C pseudo-depends on x1
	rldimi	s1, x1, 63, 0
	std	s1, 0(rp)
	mr	r3, r11
	blr

L(n2):	addi	rp, rp, -8
	srdi	s0, x1, 1
	rldicl	r11, x1, 0, 63		C return value
	b	L(cj2)
EPILOGUE()
