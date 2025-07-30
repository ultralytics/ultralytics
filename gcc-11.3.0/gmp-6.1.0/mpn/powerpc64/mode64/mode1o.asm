dnl  PowerPC-64 mpn_modexact_1_odd -- mpn by limb exact remainder.

dnl  Copyright 2006 Free Software Foundation, Inc.

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

C                  cycles/limb
C POWER3/PPC630        13-19
C POWER4/PPC970         16
C POWER5                16
C POWER6                 ?
C POWER7                12

C TODO
C  * Check if n=1 code is really an improvement.  It probably isn't.
C  * Make more similar to dive_1.asm.

C INPUT PARAMETERS
define(`up', `r3')
define(`n',  `r4')
define(`d',  `r5')
define(`cy', `r6')


ASM_START()

EXTERN(binvert_limb_table)

PROLOGUE(mpn_modexact_1c_odd,toc)
	addic.	n, n, -1		C set carry as side effect
	ld	r8, 0(up)
	bne	cr0, L(2)
	cmpld	cr7, r6, r8
	bge	cr7, L(4)
	subf	r8, r6, r8
	divdu	r3, r8, d
	mulld	r3, r3, d
	subf.	r3, r3, r8
	beqlr	cr0
	subf	r3, r3, d
	blr

L(4):	subf	r3, r8, r6
	divdu	r8, r3, d
	mulld	r8, r8, d
	subf	r3, r8, r3
	blr

L(2):	LEA(	r7, binvert_limb_table)
	rldicl	r9, d, 63, 57
	mtctr	n
	lbzx	r0, r7, r9
	mulld	r7, r0, r0
	sldi	r0, r0, 1
	mulld	r7, d, r7
	subf	r0, r7, r0
	mulld	r9, r0, r0
	sldi	r0, r0, 1
	mulld	r9, d, r9
	subf	r0, r9, r0
	mulld	r7, r0, r0
	sldi	r0, r0, 1
	mulld	r7, d, r7
	subf	r9, r7, r0

	ALIGN(16)
L(loop):
	subfe	r0, r6, r8
	ld	r8, 8(up)
	addi	up, up, 8
	mulld	r0, r9, r0
	mulhdu	r6, r0, d
	bdnz	L(loop)

	cmpld	cr7, d, r8
	blt	cr7, L(10)

	subfe	r0, r0, r0
	subf	r6, r0, r6
	cmpld	cr7, r6, r8
	subf	r3, r8, r6
	bgelr	cr7
	add	r3, d, r3
	blr

L(10):	subfe	r0, r6, r8
	mulld	r0, r9, r0
	mulhdu	r3, r0, d
	blr
EPILOGUE()
ASM_END()
