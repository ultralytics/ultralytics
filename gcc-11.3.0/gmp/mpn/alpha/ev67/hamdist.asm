dnl  Alpha ev67 mpn_hamdist -- mpn hamming distance.

dnl  Copyright 2003, 2005 Free Software Foundation, Inc.

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


C ev67: 2.5 cycles/limb


C unsigned long mpn_hamdist (mp_srcptr xp, mp_srcptr yp, mp_size_t size);
C
C The hope was for 2.0 c/l here, but that isn't achieved.  We're limited by
C renaming register shortage.  Since we need 5 instructions per limb, further
C unrolling could approach 1.5 c/l.
C
C The main loop processes two limbs from each operand on each iteration.  An
C odd size is handled by processing xp[0]^yp[0] at the start.  If the size
C is even that result is discarded, and is repeated by the main loop.
C

ASM_START()
PROLOGUE(mpn_hamdist)

	C r16	xp
	C r17	yp
	C r18	size

	ldq	r1, 0(r16)		C L0  xp[0]
	ldq	r2, 0(r17)		C L1  yp[0]
	and	r18, 1, r8		C U1  1 if size odd
	srl	r18, 1, r18		C U0  size, limb pairs

	clr	r0			C L0  initial total
	s8addq	r8, r17, r17		C U1  yp++ if size odd
	s8addq	r8, r16, r16		C L1  xp++ if size odd
	clr	r6			C U0  dummy initial xor 1

	xor	r1, r2, r5		C L   initial xor 0
	beq	r18, L(one)		C U   if size==1

	cmoveq	r8, r31, r5		C L   discard first limb if size even
	unop				C U


	ALIGN(16)
L(top):
	C r0	total accumulating
	C r7	xor 0
	C r8	xor 1
	C r16	xp, incrementing
	C r17	yp, incrementing
	C r18	size, limb pairs, decrementing

	ldq	r1, 0(r16)		C L
	ldq	r2, 0(r17)		C L
	ctpop	r5, r7			C U0
	lda	r16, 16(r16)		C U

	ldq	r3, -8(r16)		C L
	ldq	r4, 8(r17)		C L
	ctpop	r6, r8			C U0
	lda	r17, 16(r17)		C U

	ldl	r31, 256(r16)		C L	prefetch
	ldl	r31, 256(r17)		C L	prefetch
	xor	r1, r2, r5		C U
	lda	r18, -1(r18)		C U

	xor	r3, r4, r6		C U
	addq	r0, r7, r0		C L
	addq	r0, r8, r0		C L
	bne	r18, L(top)		C U


	ctpop	r6, r8			C U0
	addq	r0, r8, r0		C L
L(one):
	ctpop	r5, r7			C U0
	addq	r0, r7, r0		C L

	ret	r31, (r26), 1		C L0

EPILOGUE()
ASM_END()
