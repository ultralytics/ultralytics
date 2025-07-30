dnl  PowerPC-64 mpn_and_n, mpn_andn_n, mpn_nand_n, mpn_ior_n, mpn_iorn_n,
dnl  mpn_nior_n, mpn_xor_n, mpn_xnor_n -- mpn bitwise logical operations.

dnl  Copyright 2003-2005 Free Software Foundation, Inc.

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
C POWER3/PPC630          1.75
C POWER4/PPC970          2.10
C POWER5                 ?
C POWER6                 ?
C POWER7                 1.75

C   n	   POWER3/PPC630   POWER4/PPC970
C     1	       15.00	       15.33
C     2		7.50		7.99
C     3		5.33		6.00
C     4		4.50		4.74
C     5		4.20		4.39
C     6		3.50		3.99
C     7		3.14		3.64
C     8		3.00		3.36
C     9		3.00		3.36
C    10		2.70		3.25
C    11		2.63		3.11
C    12		2.58		3.00
C    13		2.61		3.02
C    14		2.42		2.82
C    15		2.40		2.79
C    50		2.08		2.67
C   100		1.85		2.31
C   200		1.80		2.18
C   400		1.77		2.14
C  1000		1.76		2.10#
C  2000		1.75#		2.13
C  4000		2.30		2.57
C  8000		2.62		2.58
C 16000		2.52		4.25
C 32000		2.49	       16.25
C 64000		2.66	       18.76

ifdef(`OPERATION_and_n',
`	define(`func',`mpn_and_n')
	define(`logop',		`and')')
ifdef(`OPERATION_andn_n',
`	define(`func',`mpn_andn_n')
	define(`logop',		`andc')')
ifdef(`OPERATION_nand_n',
`	define(`func',`mpn_nand_n')
	define(`logop',		`nand')')
ifdef(`OPERATION_ior_n',
`	define(`func',`mpn_ior_n')
	define(`logop',		`or')')
ifdef(`OPERATION_iorn_n',
`	define(`func',`mpn_iorn_n')
	define(`logop',		`orc')')
ifdef(`OPERATION_nior_n',
`	define(`func',`mpn_nior_n')
	define(`logop',		`nor')')
ifdef(`OPERATION_xor_n',
`	define(`func',`mpn_xor_n')
	define(`logop',		`xor')')
ifdef(`OPERATION_xnor_n',
`	define(`func',`mpn_xnor_n')
	define(`logop',		`eqv')')

C INPUT PARAMETERS
C rp	r3
C up	r4
C vp	r5
C n	r6

MULFUNC_PROLOGUE(mpn_and_n mpn_andn_n mpn_nand_n mpn_ior_n mpn_iorn_n mpn_nior_n mpn_xor_n mpn_xnor_n)

ASM_START()
PROLOGUE(func)
	ld	r8, 0(r4)	C read lowest u limb
	ld	r9, 0(r5)	C read lowest v limb
	addi	r6, r6, 3	C compute branch count (1)
	rldic.	r0, r6, 3, 59	C r0 = (n-1 & 3) << 3; cr0 = (n == 4(t+1))?
	cmpldi	cr6, r0, 16	C cr6 = (n cmp 4t + 3)

ifdef(`HAVE_ABI_mode32',
`	rldicl	r6, r6, 62,34',	C ...branch count
`	rldicl	r6, r6, 62, 2')	C ...branch count
	mtctr	r6

	ld	r6, 0(r4)	C read lowest u limb (again)
	ld	r7, 0(r5)	C read lowest v limb (again)

	add	r5, r5, r0	C offset vp
	add	r4, r4, r0	C offset up
	add	r3, r3, r0	C offset rp

	beq	cr0, L(L01)
	blt	cr6, L(L10)
	beq	cr6, L(L11)
	b	L(L00)

L(oop):	ld	r8, -24(r4)
	ld	r9, -24(r5)
	logop	r10, r6, r7
	std	r10, -32(r3)
L(L00):	ld	r6, -16(r4)
	ld	r7, -16(r5)
	logop	r10, r8, r9
	std	r10, -24(r3)
L(L11):	ld	r8, -8(r4)
	ld	r9, -8(r5)
	logop	r10, r6, r7
	std	r10, -16(r3)
L(L10):	ld	r6, 0(r4)
	ld	r7, 0(r5)
	logop	r10, r8, r9
	std	r10, -8(r3)
L(L01):	addi	r5, r5, 32
	addi	r4, r4, 32
	addi	r3, r3, 32
	bdnz	L(oop)

	logop	r10, r6, r7
	std	r10, -32(r3)
	blr
EPILOGUE()
