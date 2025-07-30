dnl  S/390-32 mpn_copyd

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
C            cycles/limb
C z900		 1.65
C z990           1.125
C z9		 ?
C z10		 ?
C z196		 ?

C FIXME:
C  * Avoid saving/restoring callee-saves registers for n < 3.  This could be
C    done by setting rp=r1, up=r2, i=r0 and r3,r4,r5 for clock regs.
C    We could then use r3...r10 in main loop.

C INPUT PARAMETERS
define(`rp_param',	`%r2')
define(`up_param',	`%r3')
define(`n',		`%r4')

define(`rp',	`%r8')
define(`up',	`%r9')

ASM_START()
PROLOGUE(mpn_copyd)
	stm	%r6, %r11, 24(%r15)

	lr	%r1, n
	sll	%r1, 2
	la	%r10, 8(n)
	ahi	%r1, -32
	srl	%r10, 3
	lhi	%r11, -32

	la	rp, 0(%r1,rp_param)	C FIXME use lay on z990 and later
	la	up, 0(%r1,up_param)	C FIXME use lay on z990 and later

	lhi	%r7, 7
	nr	%r7, n			C n mod 8
	chi	%r7, 2
	jh	L(b34567)
	chi	%r7, 1
	je	L(b1)
	jh	L(b2)

L(b0):	brct	%r10, L(top)
	j	L(end)

L(b1):	l	%r0, 28(up)
	ahi	up, -4
	st	%r0, 28(rp)
	ahi	rp, -4
	brct	%r10, L(top)
	j	L(end)

L(b2):	lm	%r0, %r1, 24(up)
	ahi	up, -8
	stm	%r0, %r1, 24(rp)
	ahi	rp, -8
	brct	%r10, L(top)
	j	L(end)

L(b34567):
	chi	%r7, 4
	jl	L(b3)
	je	L(b4)
	chi	%r7, 6
	je	L(b6)
	jh	L(b7)

L(b5):	lm	%r0, %r4, 12(up)
	ahi	up, -20
	stm	%r0, %r4, 12(rp)
	ahi	rp, -20
	brct	%r10, L(top)
	j	L(end)

L(b3):	lm	%r0, %r2, 20(up)
	ahi	up, -12
	stm	%r0, %r2, 20(rp)
	ahi	rp, -12
	brct	%r10, L(top)
	j	L(end)

L(b4):	lm	%r0, %r3, 16(up)
	ahi	up, -16
	stm	%r0, %r3, 16(rp)
	ahi	rp, -16
	brct	%r10, L(top)
	j	L(end)

L(b6):	lm	%r0, %r5, 8(up)
	ahi	up, -24
	stm	%r0, %r5, 8(rp)
	ahi	rp, -24
	brct	%r10, L(top)
	j	L(end)

L(b7):	lm	%r0, %r6, 4(up)
	ahi	up, -28
	stm	%r0, %r6, 4(rp)
	ahi	rp, -28
	brct	%r10, L(top)
	j	L(end)

L(top):	lm	%r0, %r7, 0(up)
	la	up, 0(%r11,up)
	stm	%r0, %r7, 0(rp)
	la	rp, 0(%r11,rp)
	brct	%r10, L(top)

L(end):	lm	%r6, %r11, 24(%r15)
	br	%r14
EPILOGUE()
