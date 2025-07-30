dnl  S/390-32 mpn_lshift.

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
C z900		 6
C z990	         3
C z9		 ?
C z10		 ?
C z196		 ?

C TODO
C  *

C INPUT PARAMETERS
define(`rp',	`%r2')
define(`up',	`%r3')
define(`n',	`%r4')
define(`cnt',	`%r5')

ASM_START()
PROLOGUE(mpn_lshift)
	lr	%r1, n
	sll	%r1, 2
	stm	%r6, %r12, 24(%r15)
	la	up, 0(%r1,up)		C put up near end of U
	la	rp, 0(%r1,rp)		C put rp near end of R
	ahi	up, -20
	ahi	rp, -16
	lhi	%r8, 32
	sr	%r8, cnt
	l	%r12, 16(up)
	srl	%r12, 0(%r8)		C return value
	lhi	%r7, 3
	nr	%r7, n
	srl	n, 2
	je	L(b0)
	chi	%r7, 2
	jl	L(b1)
	je	L(b2)

L(b3):	l	%r10, 16(up)
	l	%r11, 12(up)
	l	%r9,   8(up)
	ahi	up, -8
	lr	%r8, %r11
	sldl	%r10, 0(cnt)
	sldl	%r8,  0(cnt)
	st	%r10, 12(rp)
	st	%r8,   8(rp)
	ahi	rp, -8
	ltr	n, n
	je	L(end)
	j	L(top)

L(b2):	l	%r10, 16(up)
	l	%r11, 12(up)
	ahi	up, -4
	sldl	%r10, 0(cnt)
	st	%r10, 12(rp)
	ahi	rp, -4
	ltr	n, n
	je	L(end)
	j	L(top)

L(b1):	ltr	n, n
	je	L(end)
	j	L(top)

L(b0):	l	%r10,16(up)
	l	%r8, 12(up)
	l	%r6,  8(up)
	l	%r0,  4(up)
	ahi	up, -12
	lr	%r11, %r8
	lr	%r9,  %r6
	lr	%r7,  %r0
	sldl	%r10,0(cnt)
	sldl	%r8, 0(cnt)
	sldl	%r6, 0(cnt)
	st	%r10, 12(rp)
	st	%r8,   8(rp)
	st	%r6,   4(rp)
	ahi	rp, -12
	ahi	n, -1
	je	L(end)

	ALIGN(8)
L(top):	l	%r10, 16(up)
	l	%r8,  12(up)
	l	%r6,   8(up)
	l	%r0,   4(up)
	l	%r1,   0(up)
	lr	%r11, %r8
	lr	%r9,  %r6
	lr	%r7,  %r0
	ahi	up, -16
	sldl	%r10, 0(cnt)
	sldl	%r8,  0(cnt)
	sldl	%r6,  0(cnt)
	sldl	%r0,  0(cnt)
	st	%r10, 12(rp)
	st	%r8,   8(rp)
	st	%r6,   4(rp)
	st	%r0,   0(rp)
	ahi	rp, -16
	brct	n, L(top)

L(end):	l	%r10, 16(up)
	sll	%r10, 0(cnt)
	st	%r10, 12(rp)

	lr	%r2, %r12
	lm	%r6, %r12, 24(%r15)
	br	%r14
EPILOGUE()
