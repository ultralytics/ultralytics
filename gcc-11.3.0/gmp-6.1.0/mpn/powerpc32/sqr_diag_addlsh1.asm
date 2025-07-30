dnl  PowerPC-32 mpn_sqr_diag_addlsh1.

dnl  Copyright 2013 Free Software Foundation, Inc.

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

C                cycles/limb
C 603e			 ?
C 604e			 ?
C 75x (G3)		 ?
C 7400,7410 (G4)	 ?
C 744x,745x (G4+)	 6
C power4/ppc970		 ?
C power5		 ?

C This has been feebly optimised for 7447 but not for any other CPU.

define(`rp',	r3)
define(`tp',	r4)
define(`up',	r5)
define(`n',	r6)

ASM_START()
PROLOGUE(mpn_sqr_diag_addlsh1)
	addi	n, n, -1
	addi	tp, tp, -4
	mtctr	n
	lwz	r0, 0(up)
	li	r10, 0
	mullw	r7, r0, r0
	stw	r7, 0(rp)
	mulhwu	r6, r0, r0
	addic	r31, r31, 0	C clear CF

	ALIGN(16)
L(top):	lwzu	r0, 4(up)
	mullw	r7, r0, r0
	lwz	r8, 4(tp)
	lwzu	r9, 8(tp)
	rlwimi	r10, r8, 1,0,30
	srwi	r11, r8, 31
	rlwimi	r11, r9, 1,0,30
	adde	r10, r10, r6
	adde	r11, r11, r7
	stw	r10, 4(rp)
	srwi	r10, r9, 31
	mulhwu	r6, r0, r0
	stwu	r11, 8(rp)
	bdnz	L(top)

	adde	r10, r10, r6
	stw	r10, 4(rp)
	blr
EPILOGUE()
