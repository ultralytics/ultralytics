dnl  Alpha ev67 mpn_gcd_1 -- Nx1 greatest common divisor.

dnl  Copyright 2003, 2004 Free Software Foundation, Inc.

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


C ev67: 3.4 cycles/bitpair for 1x1 part


C mp_limb_t mpn_gcd_1 (mp_srcptr xp, mp_size_t xsize, mp_limb_t y);
C
C In the 1x1 part, the algorithm is to change x,y to abs(x-y),min(x,y) and
C strip trailing zeros from abs(x-y) to maintain x and y both odd.
C
C The trailing zeros are calculated from just x-y, since in twos-complement
C there's the same number of trailing zeros on d or -d.  This means the cttz
C runs in parallel with abs(x-y).
C
C The loop takes 5 cycles, and at 0.68 iterations per bit for two N-bit
C operands with this algorithm gives the measured 3.4 c/l.
C
C The slottings shown are for SVR4 style systems, Unicos differs in the
C initial gp setup and the LEA.
C
C Enhancement:
C
C On the jsr, !lituse_jsr! (when available) would allow the linker to relax
C it to a bsr, but probably only in a static binary.  Plain "jsr foo" gives
C the right object code for relaxation, and ought to be available
C everywhere, but we prefer to schedule the GOT ldq (LEA) back earlier, for
C the usual case of running in a shared library.
C
C bsr could perhaps be used explicitly anyway.  We should be able to assume
C modexact is in the same module as us (ie. shared library or mainline).
C Would there be any worries about the size of the displacement?  Could
C always put modexact and gcd_1 in the same .o to be certain.

ASM_START()
PROLOGUE(mpn_gcd_1, gp)

	C r16	xp
	C r17	size
	C r18	y

	C ldah				C l
	C lda				C u

	ldq	r0, 0(r16)		C L   x = xp[0]
	lda	r30, -32(r30)		C u   alloc stack

	LEA(  r27, mpn_modexact_1c_odd)	C L   modexact addr, ldq (gp)
	stq	r10, 16(r30)		C L   save r10
	cttz	r18, r10		C U0  y twos
	cmpeq	r17, 1, r5		C u   test size==1

	stq	r9, 8(r30)		C L   save r9
	clr	r19			C u   zero c for modexact
	unop
	unop

	cttz	r0, r6			C U0  x twos
	stq	r26, 0(r30)		C L   save ra

	srl	r18, r10, r18		C U   y odd

	mov	r18, r9			C l   hold y across call

	cmpult	r6, r10, r2		C u   test x_twos < y_twos

	cmovne	r2, r6, r10		C l   common_twos = min(x_twos,y_twos)
	bne	r5, L(one)		C U   no modexact if size==1
	jsr	r26, (r27), mpn_modexact_1c_odd   C L0

	LDGP(	r29, 0(r26))		C u,l ldah,lda
	cttz	r0, r6			C U0  new x twos
	ldq	r26, 0(r30)		C L   restore ra

L(one):
	mov	r9, r1			C u   y
	ldq	r9, 8(r30)		C L   restore r9
	mov	r10, r2			C u   common twos
	ldq	r10, 16(r30)		C L   restore r10

	lda	r30, 32(r30)		C l   free stack
	beq	r0, L(done)		C U   return y if x%y==0

	srl	r0, r6, r0		C U   x odd
	unop

	ALIGN(16)
L(top):
	C r0	x
	C r1	y
	C r2	common twos, for use at end

	subq	r0, r1, r7		C l0  d = x - y
	cmpult	r0, r1, r16		C u0  test x >= y

	subq	r1, r0, r4		C l0  new_x = y - x
	cttz	r7, r8			C U0  d twos

	cmoveq	r16, r7, r4		C l0  new_x = d if x>=y
	cmovne	r16, r0, r1		C u0  y = x if x<y
	unop				C l   \ force cmoveq into l0
	unop				C u   /

	C				C cmoveq2 L0, cmovne2 U0

	srl	r4, r8, r0		C U0  x = new_x >> twos
	bne	r7, L(top)		C U1  stop when d==0


L(done):
	sll	r1, r2, r0		C U0  return y << common_twos
	ret	r31, (r26), 1		C L0

EPILOGUE()
ASM_END()
