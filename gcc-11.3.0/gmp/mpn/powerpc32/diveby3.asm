dnl  PowerPC-32 mpn_divexact_by3 -- mpn by 3 exact division

dnl  Copyright 2002, 2003, 2005, 2006 Free Software Foundation, Inc.

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
C 603e:              ?
C 604e:              5
C 75x (G3):          ?
C 7400,7410 (G4):    8
C 744x,745x (G4+):   6
C power4/ppc970:    12
C power5:            ?

C void mpn_divexact_by3 (mp_ptr dst, mp_srcptr src, mp_size_t size);
C
C We avoid the slow subfe instruction and instead rely on an extremely unlikely
C branch.
C
C The mullw has the inverse in the first operand, since 0xAA..AB won't allow
C any early-out.  The src[] data normally won't either, but there's at least
C a chance, whereas 0xAA..AB never will.  If, for instance, src[] is all
C zeros (not a sensible input of course) we run at 7.0 c/l on ppc750.
C
C The mulhwu has the "3" multiplier in the second operand, which lets 750 and
C 7400 use an early-out.

C INPUT PARAMETERS
define(`rp', `r3')
define(`up', `r4')
define(`n',  `r5')
define(`cy', `r6')

ASM_START()
PROLOGUE(mpn_divexact_by3c)
	lwz	r11, 0(up)
	mtctr	n
	lis	r12, 0xAAAA
	ori	r12, r12, 0xAAAB
	li	r10, 3

	cmplw	cr7, cy, r11
	subf	r11, cy, r11

	mullw	r0, r11, r12
	stw	r0, 0(rp)
	bdz	L(one)

L(top):	lwzu	r9, 4(up)
	mulhwu	r7, r0, r10
	bgt-	cr7, L(adj)		C very unlikely branch
L(bko):	cmplw	cr7, r7, r9
	subf	r0, r7, r9
	mullw	r0, r12, r0
	stwu	r0, 4(rp)
	bdnz	L(top)

L(one):	mulhwu	r3, r0, r10
	blelr+	cr7
	addi	r3, r3, 1
	blr

L(adj):	addi	r7, r7, 1
	b	L(bko)
EPILOGUE()
ASM_END()
