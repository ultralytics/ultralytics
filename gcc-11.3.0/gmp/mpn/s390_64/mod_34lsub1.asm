dnl  S/390-64 mpn_mod_34lsub1

dnl  Copyright 2011 Free Software Foundation, Inc.

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

C            cycles/limb
C z900		 5.8
C z990           2
C z9		 ?
C z10		 4.5
C z196		 ?

C TODO
C  * Optimise summation code, see x86_64.

C INPUT PARAMETERS
define(`rp',	`%r2')
define(`n',	`%r3')

ASM_START()
PROLOGUE(mpn_mod_34lsub1)
	stmg	%r7, %r12, 56(%r15)
	lghi	%r11, 0
	lghi	%r12, 0
	lghi	%r0, 0
	lghi	%r8, 0
	lghi	%r9, 0
	lghi	%r10, 0
	lghi	%r7, 0
	aghi	%r3, -3
	jl	.L3

L(top):	alg	%r0, 0(%r2)
	alcg	%r12, 8(%r2)
	alcg	%r11, 16(%r2)
	alcgr	%r8, %r7
	la	%r2, 24(%r2)
	aghi	%r3, -3
	jnl	L(top)

	lgr	%r7, %r8
	srlg	%r1, %r11, 16
	nihh	%r7, 0			C 0xffffffffffff
	agr	%r7, %r1
	srlg	%r8, %r8, 48
	agr	%r7, %r8
	sllg	%r11, %r11, 32
	nihh	%r11, 0
	agr	%r7, %r11
.L3:
	cghi	%r3, -3
	je	.L6
	alg	%r0, 0(%r2)
	alcgr	%r10, %r10
	cghi	%r3, -2
	je	.L6
	alg	%r12, 8(%r2)
	alcgr	%r9, %r9
.L6:
	srlg	%r1, %r0, 48
	nihh	%r0, 0			C 0xffffffffffff
	agr	%r0, %r1
	agr	%r0, %r7
	srlg	%r1, %r12, 32
	agr	%r0, %r1
	srlg	%r1, %r10, 32
	agr	%r0, %r1
	llgfr	%r12, %r12
	srlg	%r1, %r9, 16
	sllg	%r12, %r12, 16
	llgfr	%r10, %r10
	agr	%r0, %r1
	llill	%r2, 65535
	agr	%r0, %r12
	sllg	%r10, %r10, 16
	ngr	%r2, %r9
	agr	%r0, %r10
	sllg	%r2, %r2, 32
	agr	%r2, %r0
	lmg	%r7, %r12, 56(%r15)
	br	%r14
EPILOGUE()
