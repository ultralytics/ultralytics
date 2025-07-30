dnl  S/390-64 mpn_rshift.

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
C z900		 7
C z990           3
C z9		 ?
C z10		 6
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
PROLOGUE(mpn_rshift)
	cghi	n, 3
	jh	L(gt1)

	stmg	%r6, %r7, 48(%r15)
	larl	%r1, L(tab)-4
	lcgr	tnc, cnt
	sllg	n, n, 2
	b	0(n,%r1)
L(tab):	j	L(n1)
	j	L(n2)
	j	L(n3)

L(n1):	lg	%r1, 0(up)
	srlg	%r0, %r1, 0(cnt)
	stg	%r0, 0(rp)
	sllg	%r2, %r1, 0(tnc)
	lg	%r6, 48(%r15)		C restoring r7 not needed
	br	%r14

L(n2):	lg	%r1, 0(up)
	sllg	%r4, %r1, 0(tnc)
	srlg	%r0, %r1, 0(cnt)
	lg	%r1, 8(up)
	sllg	%r7, %r1, 0(tnc)
	ogr	%r7, %r0
	srlg	%r0, %r1, 0(cnt)
	stg	%r7, 0(rp)
	stg	%r0, 8(rp)
	lgr	%r2, %r4
	lmg	%r6, %r7, 48(%r15)
	br	%r14


L(n3):	lg	%r1, 0(up)
	sllg	%r4, %r1, 0(tnc)
	srlg	%r0, %r1, 0(cnt)
	lg	%r1, 8(up)
	sllg	%r7, %r1, 0(tnc)
	ogr	%r7, %r0
	srlg	%r0, %r1, 0(cnt)
	stg	%r7, 0(rp)
	lg	%r1, 16(up)
	sllg	%r7, %r1, 0(tnc)
	ogr	%r7, %r0
	srlg	%r0, %r1, 0(cnt)
	stg	%r7, 8(rp)
	stg	%r0, 16(rp)
	lgr	%r2, %r4
	lmg	%r6, %r7, 48(%r15)
	br	%r14

L(gt1):	stmg	%r6, %r13, 48(%r15)
	lcgr	tnc, cnt		C tnc = -cnt

	sllg	%r1, n, 3
	srlg	%r0, n, 2		C loop count

	lghi	%r7, 3
	ngr	%r7, n
	je	L(b0)
	cghi	%r7, 2
	jl	L(b1)
	je	L(b2)

L(b3):	aghi	rp, -8
	lg	%r7, 0(up)
	sllg	%r9, %r7, 0(tnc)
	srlg	%r11, %r7, 0(cnt)
	lg	%r8, 8(up)
	lg	%r7, 16(up)
	sllg	%r4, %r8, 0(tnc)
	srlg	%r13, %r8, 0(cnt)
	ogr	%r11, %r4
	la	up, 24(up)
	j	L(lm3)

L(b2):	aghi	rp, -16
	lg	%r8, 0(up)
	lg	%r7, 8(up)
	sllg	%r9, %r8, 0(tnc)
	srlg	%r13, %r8, 0(cnt)
	la	up, 16(up)
	j	L(lm2)

L(b1):	aghi	rp, -24
	lg	%r7, 0(up)
	sllg	%r9, %r7, 0(tnc)
	srlg	%r11, %r7, 0(cnt)
	lg	%r8, 8(up)
	lg	%r7, 16(up)
	sllg	%r4, %r8, 0(tnc)
	srlg	%r10, %r8, 0(cnt)
	ogr	%r11, %r4
	la	up, 8(up)
	j	L(lm1)

L(b0):	aghi	rp, -32
	lg	%r8, 0(up)
	lg	%r7, 8(up)
	sllg	%r9, %r8, 0(tnc)
	srlg	%r10, %r8, 0(cnt)
	j	L(lm0)

	ALIGN(8)
L(top):	sllg	%r4, %r8, 0(tnc)
	srlg	%r13, %r8, 0(cnt)
	ogr	%r11, %r4
	stg	%r10, 0(rp)
L(lm3):	stg	%r11, 8(rp)
L(lm2):	sllg	%r12, %r7, 0(tnc)
	srlg	%r11, %r7, 0(cnt)
	lg	%r8, 0(up)
	lg	%r7, 8(up)
	ogr	%r13, %r12
	sllg	%r4, %r8, 0(tnc)
	srlg	%r10, %r8, 0(cnt)
	ogr	%r11, %r4
	stg	%r13, 16(rp)
L(lm1):	stg	%r11, 24(rp)
L(lm0):	sllg	%r12, %r7, 0(tnc)
	aghi	rp, 32
	srlg	%r11, %r7, 0(cnt)
	lg	%r8, 16(up)
	lg	%r7, 24(up)
	aghi	up, 32
	ogr	%r10, %r12
	brctg	%r0, L(top)

L(end):	sllg	%r4, %r8, 0(tnc)
	srlg	%r13, %r8, 0(cnt)
	ogr	%r11, %r4
	stg	%r10, 0(rp)
	stg	%r11, 8(rp)
	sllg	%r12, %r7, 0(tnc)
	srlg	%r11, %r7, 0(cnt)
	ogr	%r13, %r12
	stg	%r13, 16(rp)
	stg	%r11, 24(rp)
	lgr	%r2, %r9

	lmg	%r6, %r13, 48(%r15)
	br	%r14
EPILOGUE()
