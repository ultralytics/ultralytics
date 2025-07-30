dnl  S/390-32/esame mpn_mul_basecase.

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
C z900		 ?
C z990		 ?
C z9		 ?
C z10		 ?
C z196		 ?

C TODO
C  * Perhaps add special case for un <= 2.
C  * Replace loops by faster code.  The mul_1 and addmul_1 loops could be sped
C    up by about 10%.

C INPUT PARAMETERS
define(`rp',	`%r2')
define(`up',	`%r3')
define(`un',	`%r4')
define(`vp',	`%r5')
define(`vn',	`%r6')

define(`zero',	`%r8')

ASM_START()
PROLOGUE(mpn_mul_basecase)
	chi	un, 2
	jhe	L(ge2)

C un = vn = 1
	l	%r1, 0(vp)
	ml	%r0, 0(up)
	st	%r1, 0(rp)
	st	%r0, 4(rp)
	br	%r14

L(ge2):	C jne	L(gen)


L(gen):
C mul_1 =======================================================================

	stm	%r6, %r12, 24(%r15)
	lhi	zero, 0
	ahi	un, -1

	l	%r7, 0(vp)
	l	%r11, 0(up)
	lhi	%r12, 4			C init index register
	mlr	%r10, %r7
	lr	%r9, un
	st	%r11, 0(rp)
	cr	%r15, %r15		C clear carry flag

L(tm):	l	%r1, 0(%r12,up)
	mlr	%r0, %r7
	alcr	%r1, %r10
	lr	%r10, %r0		C copy high part to carry limb
	st	%r1, 0(%r12,rp)
	la	%r12, 4(%r12)
	brct	%r9, L(tm)

	alcr	%r0, zero
	st	%r0, 0(%r12,rp)

C addmul_1 loop ===============================================================

	ahi	vn, -1
	je	L(outer_end)
L(outer_loop):

	la	rp, 4(rp)		C rp += 1
	la	vp, 4(vp)		C up += 1
	l	%r7, 0(vp)
	l	%r11, 0(up)
	lhi	%r12, 4			C init index register
	mlr	%r10, %r7
	lr	%r9, un
	al	%r11, 0(rp)
	st	%r11, 0(rp)

L(tam):	l	%r1, 0(%r12,up)
	l	%r11, 0(%r12,rp)
	mlr	%r0, %r7
	alcr	%r1, %r11
	alcr	%r0, zero
	alr	%r1, %r10
	lr	%r10, %r0
	st	%r1, 0(%r12,rp)
	la	%r12, 4(%r12)
	brct	%r9, L(tam)

	alcr	%r0, zero
	st	%r0, 0(%r12,rp)

	brct	vn, L(outer_loop)
L(outer_end):

	lm	%r6, %r12, 24(%r15)
	br	%r14
EPILOGUE()
