dnl  S/390-32 mpn_sqr_basecase.

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
C z10		 ?
C z196		 ?

C TODO
C  * Clean up.
C  * Stop iterating addmul_1 loop at latest for n = 2, implement longer tail.
C    This will ask for basecase handling of n = 3.
C  * Update counters and pointers more straightforwardly, possibly lowering
C    register usage.
C  * Should we use this allocation-free style for more sqr_basecase asm
C    implementations?  The only disadvantage is that it requires R != U.
C  * Replace loops by faster code.  The mul_1 and addmul_1 loops could be sped
C    up by about 10%.  The sqr_diag_addlsh1 loop could probably be sped up even
C    more.

C INPUT PARAMETERS
define(`rp',	`%r2')
define(`up',	`%r3')
define(`n',	`%r4')

define(`zero',	`%r8')
define(`rp_saved',	`%r9')
define(`up_saved',	`%r13')
define(`n_saved',	`%r14')

ASM_START()
PROLOGUE(mpn_sqr_basecase)
	ahi	n, -2
	jhe	L(ge2)

C n = 1
	l	%r5, 0(up)
	mlr	%r4, %r5
	st	%r5, 0(rp)
	st	%r4, 4(rp)
	br	%r14

L(ge2):	jne	L(gen)

C n = 2
	stm	%r6, %r8, 24(%r15)
	lhi	zero, 0

	l	%r5, 0(up)
	mlr	%r4, %r5		C u0 * u0
	l	%r1, 4(up)
	mlr	%r0, %r1		C u1 * u1
	st	%r5, 0(rp)

	l	%r7, 0(up)
	ml	%r6, 4(up)		C u0 * u1
	alr	%r7, %r7
	alcr	%r6, %r6
	alcr	%r0, zero

	alr	%r4, %r7
	alcr	%r1, %r6
	alcr	%r0, zero
	st	%r4, 4(rp)
	st	%r1, 8(rp)
	st	%r0, 12(rp)

	lm	%r6, %r8, 24(%r15)
	br	%r14

L(gen):
C mul_1 =======================================================================

	stm	%r6, %r14, 24(%r15)
	lhi	zero, 0
	lr	up_saved, up
	lr	rp_saved, rp
	lr	n_saved, n

	l	%r6, 0(up)
	l	%r11, 4(up)
	lhi	%r12, 8		C init index register
	mlr	%r10, %r6
	lr	%r5, n
	st	%r11, 4(rp)
	cr	%r15, %r15		C clear carry flag

L(tm):	l	%r1, 0(%r12,up)
	mlr	%r0, %r6
	alcr	%r1, %r10
	lr	%r10, %r0		C copy high part to carry limb
	st	%r1, 0(%r12,rp)
	la	%r12, 4(%r12)
	brct	%r5, L(tm)

	alcr	%r0, zero
	st	%r0, 0(%r12,rp)

C addmul_1 loop ===============================================================

	ahi	n, -1
	je	L(outer_end)
L(outer_loop):

	la	rp, 8(rp)		C rp += 2
	la	up, 4(up)		C up += 1
	l	%r6, 0(up)
	l	%r11, 4(up)
	lhi	%r12, 8		C init index register
	mlr	%r10, %r6
	lr	%r5, n
	al	%r11, 4(rp)
	st	%r11, 4(rp)

L(tam):	l	%r1, 0(%r12,up)
	l	%r7, 0(%r12,rp)
	mlr	%r0, %r6
	alcr	%r1, %r7
	alcr	%r0, zero
	alr	%r1, %r10
	lr	%r10, %r0
	st	%r1, 0(%r12,rp)
	la	%r12, 4(%r12)
	brct	%r5, L(tam)

	alcr	%r0, zero
	st	%r0, 0(%r12,rp)

	brct	n, L(outer_loop)
L(outer_end):

	l	%r6, 4(up)
	l	%r1, 8(up)
	lr	%r7, %r0		C Same as: l %r7, 12(,rp)
	mlr	%r0, %r6
	alr	%r1, %r7
	alcr	%r0, zero
	st	%r1, 12(rp)
	st	%r0, 16(rp)

C sqr_dia_addlsh1 ============================================================

define(`up', `up_saved')
define(`rp', `rp_saved')
	la	n, 1(n_saved)

	l	%r1, 0(up)
	mlr	%r0, %r1
	st	%r1, 0(rp)
C	clr	%r15, %r15		C clear carry (already clear per above)

L(top):	l	%r11, 4(up)
	la	up, 4(up)
	l	%r6, 4(rp)
	l	%r7, 8(rp)
	mlr	%r10, %r11
	alcr	%r6, %r6
	alcr	%r7, %r7
	alcr	%r10, zero		C propagate carry to high product limb
	alr	%r6, %r0
	alcr	%r7, %r11
	stm	%r6, %r7, 4(rp)
	la	rp, 8(rp)
	lr	%r0, %r10		C copy carry limb
	brct	n, L(top)

	alcr	%r0, zero
	st	%r0, 4(rp)

	lm	%r6, %r14, 24(%r15)
	br	%r14
EPILOGUE()
