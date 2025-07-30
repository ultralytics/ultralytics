dnl  S/390-64 mpn_lshiftc.

dnl  Copyright 2011, 2014 Free Software Foundation, Inc.

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
C z900		 9
C z990           3.5
C z9		 ?
C z10		 7
C z196		 ?

C NOTES
C  * See notes in lshift.asm.

C INPUT PARAMETERS
define(`rp',	`%r2')
define(`up',	`%r3')
define(`n',	`%r4')
define(`cnt',	`%r5')

define(`tnc',	`%r6')

ASM_START()
PROLOGUE(mpn_lshiftc)
	cghi	n, 3
	jh	L(gt1)

	stmg	%r6, %r8, 48(%r15)
	larl	%r1, L(tab)-4
	lcgr	tnc, cnt
	sllg	n, n, 2
	lghi	%r8, -1
	b	0(n,%r1)
L(tab):	j	L(n1)
	j	L(n2)
	j	L(n3)

L(n1):	lg	%r1, 0(up)
	sllg	%r0, %r1, 0(cnt)
	xgr	%r0, %r8
	stg	%r0, 0(rp)
	srlg	%r2, %r1, 0(tnc)
	lmg	%r6, %r8, 48(%r15)
	br	%r14

L(n2):	lg	%r1, 8(up)
	srlg	%r4, %r1, 0(tnc)
	sllg	%r0, %r1, 0(cnt)
	j	L(cj)

L(n3):	lg	%r1, 16(up)
	srlg	%r4, %r1, 0(tnc)
	sllg	%r0, %r1, 0(cnt)
	lg	%r1, 8(up)
	srlg	%r7, %r1, 0(tnc)
	ogr	%r7, %r0
	sllg	%r0, %r1, 0(cnt)
	xgr	%r7, %r8
	stg	%r7, 16(rp)
L(cj):	lg	%r1, 0(up)
	srlg	%r7, %r1, 0(tnc)
	ogr	%r7, %r0
	sllg	%r0, %r1, 0(cnt)
	xgr	%r7, %r8
	xgr	%r0, %r8
	stg	%r7, 8(rp)
	stg	%r0, 0(rp)
	lgr	%r2, %r4
	lmg	%r6, %r8, 48(%r15)
	br	%r14

L(gt1):	stmg	%r6, %r14, 48(%r15)
	lcgr	tnc, cnt		C tnc = -cnt

	sllg	%r1, n, 3
	srlg	%r0, n, 2		C loop count

	agr	up, %r1			C point up at end of U
	agr	rp, %r1			C point rp at end of R
	aghi	up, -56
	aghi	rp, -40

	lghi	%r7, 3
	lghi	%r14, -1
	ngr	%r7, n
	je	L(b0)
	cghi	%r7, 2
	jl	L(b1)
	je	L(b2)

L(b3):	lg	%r7, 48(up)
	srlg	%r9, %r7, 0(tnc)
	sllg	%r11, %r7, 0(cnt)
	lg	%r8, 40(up)
	lg	%r7, 32(up)
	srlg	%r4, %r8, 0(tnc)
	sllg	%r13, %r8, 0(cnt)
	ogr	%r11, %r4
	la	rp, 16(rp)
	xgr	%r11, %r14
	j	L(lm3)

L(b2):	lg	%r8, 48(up)
	lg	%r7, 40(up)
	srlg	%r9, %r8, 0(tnc)
	sllg	%r13, %r8, 0(cnt)
	la	rp, 24(rp)
	la	up, 8(up)
	j	L(lm2)

L(b1):	lg	%r7, 48(up)
	srlg	%r9, %r7, 0(tnc)
	sllg	%r11, %r7, 0(cnt)
	lg	%r8, 40(up)
	lg	%r7, 32(up)
	srlg	%r4, %r8, 0(tnc)
	sllg	%r10, %r8, 0(cnt)
	ogr	%r11, %r4
	la	rp, 32(rp)
	la	up, 16(up)
	xgr	%r11, %r14
	j	L(lm1)

L(b0):	lg	%r8, 48(up)
	lg	%r7, 40(up)
	srlg	%r9, %r8, 0(tnc)
	sllg	%r10, %r8, 0(cnt)
	la	rp, 40(rp)
	la	up, 24(up)
	j	L(lm0)

	ALIGN(8)
L(top):	srlg	%r4, %r8, 0(tnc)
	sllg	%r13, %r8, 0(cnt)
	ogr	%r11, %r4
	xgr	%r10, %r14
	xgr	%r11, %r14
	stg	%r10, 24(rp)
L(lm3):	stg	%r11, 16(rp)
L(lm2):	srlg	%r12, %r7, 0(tnc)
	sllg	%r11, %r7, 0(cnt)
	lg	%r8, 24(up)
	lg	%r7, 16(up)
	ogr	%r13, %r12
	srlg	%r4, %r8, 0(tnc)
	sllg	%r10, %r8, 0(cnt)
	ogr	%r11, %r4
	xgr	%r13, %r14
	xgr	%r11, %r14
	stg	%r13, 8(rp)
L(lm1):	stg	%r11, 0(rp)
L(lm0):	srlg	%r12, %r7, 0(tnc)
	aghi	rp, -32
	sllg	%r11, %r7, 0(cnt)
	lg	%r8, 8(up)
	lg	%r7, 0(up)
	aghi	up, -32
	ogr	%r10, %r12
	brctg	%r0, L(top)

L(end):	srlg	%r4, %r8, 0(tnc)
	sllg	%r13, %r8, 0(cnt)
	ogr	%r11, %r4
	xgr	%r10, %r14
	xgr	%r11, %r14
	stg	%r10, 24(rp)
	stg	%r11, 16(rp)
	srlg	%r12, %r7, 0(tnc)
	sllg	%r11, %r7, 0(cnt)
	ogr	%r13, %r12
	xgr	%r13, %r14
	xgr	%r11, %r14
	stg	%r13, 8(rp)
	stg	%r11, 0(rp)
	lgr	%r2, %r9

	lmg	%r6, %r14, 48(%r15)
	br	%r14
EPILOGUE()
