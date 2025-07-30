dnl  S/390-64 mpn_copyd

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
C z900		 2.67
C z990           1.5
C z9		 ?
C z10		 1.8
C z196		 ?

C FIXME:
C  * Avoid saving/restoring callee-saves registers for n < 3.  This could be
C    done by setting rp=r1, up=r2, i=r0 and r3,r4,r5 for clock regs.
C    We could then use r3...r10 in main loop.
C  * Could we use some EX trick, modifying lmg/stmg, for the feed-in code?

C INPUT PARAMETERS
define(`rp_param',	`%r2')
define(`up_param',	`%r3')
define(`n',		`%r4')

define(`rp',	`%r8')
define(`up',	`%r9')

ASM_START()
PROLOGUE(mpn_copyd)
	stmg	%r6, %r11, 48(%r15)

	sllg	%r1, n, 3
	la	%r10, 8(n)
	aghi	%r1, -64
	srlg	%r10, %r10, 3
	lghi	%r11, -64

	la	rp, 0(%r1,rp_param)	C FIXME use lay on z990 and later
	la	up, 0(%r1,up_param)	C FIXME use lay on z990 and later

	lghi	%r7, 7
	ngr	%r7, n			C n mod 8
	cghi	%r7, 2
	jh	L(b34567)
	cghi	%r7, 1
	je	L(b1)
	jh	L(b2)

L(b0):	brctg	%r10, L(top)
	j	L(end)

L(b1):	lg	%r0, 56(up)
	aghi	up, -8
	stg	%r0, 56(rp)
	aghi	rp, -8
	brctg	%r10, L(top)
	j	L(end)

L(b2):	lmg	%r0, %r1, 48(up)
	aghi	up, -16
	stmg	%r0, %r1, 48(rp)
	aghi	rp, -16
	brctg	%r10, L(top)
	j	L(end)

L(b34567):
	cghi	%r7, 4
	jl	L(b3)
	je	L(b4)
	cghi	%r7, 6
	je	L(b6)
	jh	L(b7)

L(b5):	lmg	%r0, %r4, 24(up)
	aghi	up, -40
	stmg	%r0, %r4, 24(rp)
	aghi	rp, -40
	brctg	%r10, L(top)
	j	L(end)

L(b3):	lmg	%r0, %r2, 40(up)
	aghi	up, -24
	stmg	%r0, %r2, 40(rp)
	aghi	rp, -24
	brctg	%r10, L(top)
	j	L(end)

L(b4):	lmg	%r0, %r3, 32(up)
	aghi	up, -32
	stmg	%r0, %r3, 32(rp)
	aghi	rp, -32
	brctg	%r10, L(top)
	j	L(end)

L(b6):	lmg	%r0, %r5, 16(up)
	aghi	up, -48
	stmg	%r0, %r5, 16(rp)
	aghi	rp, -48
	brctg	%r10, L(top)
	j	L(end)

L(b7):	lmg	%r0, %r6, 8(up)
	aghi	up, -56
	stmg	%r0, %r6, 8(rp)
	aghi	rp, -56
	brctg	%r10, L(top)
	j	L(end)

L(top):	lmg	%r0, %r7, 0(up)
	la	up, 0(%r11,up)
	stmg	%r0, %r7, 0(rp)
	la	rp, 0(%r11,rp)
	brctg	%r10, L(top)

L(end):	lmg	%r6, %r11, 48(%r15)
	br	%r14
EPILOGUE()
