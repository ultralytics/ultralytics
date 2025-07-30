dnl  S/390-64 mpn_mul_basecase.

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
C z990		23
C z9		 ?
C z10		28
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
	cghi	un, 2
	jhe	L(ge2)

C un = vn = 1
	lg	%r1, 0(vp)
	mlg	%r0, 0(up)
	stg	%r1, 0(rp)
	stg	%r0, 8(rp)
	br	%r14

L(ge2):	C jne	L(gen)


L(gen):
C mul_1 =======================================================================

	stmg	%r6, %r12, 48(%r15)
	lghi	zero, 0
	aghi	un, -1

	lg	%r7, 0(vp)
	lg	%r11, 0(up)
	lghi	%r12, 8			C init index register
	mlgr	%r10, %r7
	lgr	%r9, un
	stg	%r11, 0(rp)
	cr	%r15, %r15		C clear carry flag

L(tm):	lg	%r1, 0(%r12,up)
	mlgr	%r0, %r7
	alcgr	%r1, %r10
	lgr	%r10, %r0		C copy high part to carry limb
	stg	%r1, 0(%r12,rp)
	la	%r12, 8(%r12)
	brctg	%r9, L(tm)

	alcgr	%r0, zero
	stg	%r0, 0(%r12,rp)

C addmul_1 loop ===============================================================

	aghi	vn, -1
	je	L(outer_end)
L(outer_loop):

	la	rp, 8(rp)		C rp += 1
	la	vp, 8(vp)		C up += 1
	lg	%r7, 0(vp)
	lg	%r11, 0(up)
	lghi	%r12, 8			C init index register
	mlgr	%r10, %r7
	lgr	%r9, un
	alg	%r11, 0(rp)
	stg	%r11, 0(rp)

L(tam):	lg	%r1, 0(%r12,up)
	lg	%r11, 0(%r12,rp)
	mlgr	%r0, %r7
	alcgr	%r1, %r11
	alcgr	%r0, zero
	algr	%r1, %r10
	lgr	%r10, %r0
	stg	%r1, 0(%r12,rp)
	la	%r12, 8(%r12)
	brctg	%r9, L(tam)

	alcgr	%r0, zero
	stg	%r0, 0(%r12,rp)

	brctg	vn, L(outer_loop)
L(outer_end):

	lmg	%r6, %r12, 48(%r15)
	br	%r14
EPILOGUE()
