dnl  PowerPC-64 mpn_popcount.

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
C POWER7                 2

define(`up', r3)
define(`n',  r4)

ASM_START()
PROLOGUE(mpn_popcount)
	addi	r0, n, 1
ifdef(`HAVE_ABI_mode32',
`	rldicl	r0, r0, 63,33',	C ...branch count
`	srdi	r0, r0, 1')	C ...for ctr
	mtctr	r0

	andi.	r0, n, 1

	li	r0, 0
	li	r12, 0
	beq	L(evn)

L(odd):	ld	r4, 0(up)
	addi	up, up, 8
	popcntd	r0, r4
	bdz	L(e1)

L(evn):	ld	r4, 0(up)
	ld	r5, 8(up)
	popcntd	r8, r4
	popcntd	r9, r5
	bdz	L(e2)

	ld	r4, 16(up)
	ld	r5, 24(up)
	bdz	L(e4)
	addi	up, up, 32

L(top):	add	r0, r0, r8
	popcntd	r8, r4
	ld	r4, 0(up)
	add	r12, r12, r9
	popcntd	r9, r5
	ld	r5, 8(up)
	addi	up, up, 16
	bdnz	L(top)

L(e4):	add	r0, r0, r8
	popcntd	r8, r4
	add	r12, r12, r9
	popcntd	r9, r5
L(e2):	add	r0, r0, r8
	add	r12, r12, r9
L(e1):	add	r3, r0, r12
	blr
EPILOGUE()
