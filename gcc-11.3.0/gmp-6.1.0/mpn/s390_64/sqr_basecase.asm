dnl  S/390-64 mpn_sqr_basecase.

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
	aghi	n, -2
	jhe	L(ge2)

C n = 1
	lg	%r5, 0(up)
	mlgr	%r4, %r5
	stg	%r5, 0(rp)
	stg	%r4, 8(rp)
	br	%r14

L(ge2):	jne	L(gen)

C n = 2
	stmg	%r6, %r8, 48(%r15)
	lghi	zero, 0

	lg	%r5, 0(up)
	mlgr	%r4, %r5		C u0 * u0
	lg	%r1, 8(up)
	mlgr	%r0, %r1		C u1 * u1
	stg	%r5, 0(rp)

	lg	%r7, 0(up)
	mlg	%r6, 8(up)		C u0 * u1
	algr	%r7, %r7
	alcgr	%r6, %r6
	alcgr	%r0, zero

	algr	%r4, %r7
	alcgr	%r1, %r6
	alcgr	%r0, zero
	stg	%r4, 8(rp)
	stg	%r1, 16(rp)
	stg	%r0, 24(rp)

	lmg	%r6, %r8, 48(%r15)
	br	%r14

L(gen):
C mul_1 =======================================================================

	stmg	%r6, %r14, 48(%r15)
	lghi	zero, 0
	lgr	up_saved, up
	lgr	rp_saved, rp
	lgr	n_saved, n

	lg	%r6, 0(up)
	lg	%r11, 8(up)
	lghi	%r12, 16		C init index register
	mlgr	%r10, %r6
	lgr	%r5, n
	stg	%r11, 8(rp)
	cr	%r15, %r15		C clear carry flag

L(tm):	lg	%r1, 0(%r12,up)
	mlgr	%r0, %r6
	alcgr	%r1, %r10
	lgr	%r10, %r0		C copy high part to carry limb
	stg	%r1, 0(%r12,rp)
	la	%r12, 8(%r12)
	brctg	%r5, L(tm)

	alcgr	%r0, zero
	stg	%r0, 0(%r12,rp)

C addmul_1 loop ===============================================================

	aghi	n, -1
	je	L(outer_end)
L(outer_loop):

	la	rp, 16(rp)		C rp += 2
	la	up, 8(up)		C up += 1
	lg	%r6, 0(up)
	lg	%r11, 8(up)
	lghi	%r12, 16		C init index register
	mlgr	%r10, %r6
	lgr	%r5, n
	alg	%r11, 8(rp)
	stg	%r11, 8(rp)

L(tam):	lg	%r1, 0(%r12,up)
	lg	%r7, 0(%r12,rp)
	mlgr	%r0, %r6
	alcgr	%r1, %r7
	alcgr	%r0, zero
	algr	%r1, %r10
	lgr	%r10, %r0
	stg	%r1, 0(%r12,rp)
	la	%r12, 8(%r12)
	brctg	%r5, L(tam)

	alcgr	%r0, zero
	stg	%r0, 0(%r12,rp)

	brctg	n, L(outer_loop)
L(outer_end):

	lg	%r6, 8(up)
	lg	%r1, 16(up)
	lgr	%r7, %r0		C Same as: lg %r7, 24(,rp)
	mlgr	%r0, %r6
	algr	%r1, %r7
	alcgr	%r0, zero
	stg	%r1, 24(rp)
	stg	%r0, 32(rp)

C sqr_diag_addlsh1 ============================================================

define(`up', `up_saved')
define(`rp', `rp_saved')
	la	n, 1(n_saved)

	lg	%r1, 0(up)
	mlgr	%r0, %r1
	stg	%r1, 0(rp)
C	clr	%r15, %r15		C clear carry (already clear per above)

L(top):	lg	%r11, 8(up)
	la	up, 8(up)
	lg	%r6, 8(rp)
	lg	%r7, 16(rp)
	mlgr	%r10, %r11
	alcgr	%r6, %r6
	alcgr	%r7, %r7
	alcgr	%r10, zero		C propagate carry to high product limb
	algr	%r6, %r0
	alcgr	%r7, %r11
	stmg	%r6, %r7, 8(rp)
	la	rp, 16(rp)
	lgr	%r0, %r10		C copy carry limb
	brctg	n, L(top)

	alcgr	%r0, zero
	stg	%r0, 8(rp)

	lmg	%r6, %r14, 48(%r15)
	br	%r14
EPILOGUE()
