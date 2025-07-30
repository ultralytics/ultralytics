dnl  PowerPC-32 mpn_modexact_1_odd -- mpn by limb exact remainder.

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
C 603e:             ?
C 604e:             6.0
C 75x (G3):         6.0-13.0, depending on divisor
C 7400,7410 (G4):   6.0-13.0, depending on divisor
C 744x,745x (G4+):  8.0-10.0, depending on divisor
C power4/ppc970:   12.0
C power5:          12.0


C mp_limb_t mpn_modexact_1_odd (mp_srcptr src, mp_size_t size,
C                               mp_limb_t divisor);
C mp_limb_t mpn_modexact_1c_odd (mp_srcptr src, mp_size_t size,
C                                mp_limb_t divisor, mp_limb_t carry);
C
C For PIC, the inverse is established arithmetically since it measures about
C 5 cycles faster than the nonsense needed to access binvert_limb_table in
C SVR4 or Darwin style PIC.  AIX might be better, since it avoids bl/mflr to
C get at the GOT/TOC/whatever.
C
C Using divwu for size==1 measured about 10 cycles slower on 604e, or about
C 3-5 cycles faster on 750.  For now it doesn't seem worth bothering with.
C
C The loop allows an early-out on mullw for the inverse, and on mulhwu for
C the divisor.  So the fastest is for instance divisor==1 (inverse==-1), and
C the slowest is anything giving a full 32-bits in both, such as
C divisor==0xDEADBEEF (inverse==0x904B300F).  These establish the stated
C range above for 750 and 7400.


ASM_START()

EXTERN(binvert_limb_table)

PROLOGUE(mpn_modexact_1_odd)
	li	r6, 0

PROLOGUE(mpn_modexact_1c_odd)

	mtctr	r4			C size

ifdef(`PIC_SLOW',`
C Load from our table with PIC is so slow on Linux and Darwin that we avoid it
	rlwinm	r7, r5, 1,28,28		C (divisor << 1) & 8
	rlwinm	r8, r5, 2,28,28		C (divisor << 2) & 8
	xor	r7, r7, r8		C ((divisor << 1) ^ (divisor << 2)) & 8
	rlwinm	r4, r5, 0,28,31		C divisor low 4 bits, speedup mullw
	xor	r4, r4, r7		C inverse, 4 bits
	mullw	r7, r4, r4		C i*i
	slwi	r4, r4, 1		C 2*i
	rlwinm	r8, r5, 0,24,31		C divisor low 8 bits, speedup mullw
	mullw	r7, r7, r8		C i*i*d
	sub	r4, r4, r7		C inverse, 8 bits
',`
	LEA(	r7, binvert_limb_table)
	rlwinm	r4, r5, 31,25,31	C (divisor/2) & 0x7F
	lbzx	r4, r4,r7		C inverse, 8 bits
')

	mullw	r7, r4, r4		C i*i
	slwi	r4, r4, 1		C 2*i
	mullw	r7, r5, r7		C i*i*d   [i*i is 16 bits, so second operand]
	sub	r4, r4, r7		C inverse, 16 bits
	mullw	r7, r4, r4		C i*i
	slwi	r4, r4, 1		C 2*i
	mullw	r7, r7, r5		C i*i*d
	lwz	r0, 0(r3)		C src[0]
	sub	r4, r4, r7		C inverse, 32 bits
	subfc	r7, r6, r0		C l = src[0] - carry

	mullw	r7, r7, r4		C q = l * inverse
	bdz	L(one)

	lwzu	r0, 4(r3)		C src[1]
	mulhwu	r6, r7, r5		C carry = high(q*divisor)
	subfe	r7, r6, r0		C l = src[1] - carry
	bdz	L(two)

L(top):
	mullw	r7, r7, r4		C q = l * inverse
	lwzu	r0, 4(r3)		C src[i]
	mulhwu	r6, r7, r5		C carry = high(q*divisor)
	subfe	r7, r6, r0		C l = src[i] - carry
	bdnz	L(top)

L(two):	mullw	r7, r7, r4		C q = l * inverse
L(one):	subfe	r3, r3, r3		C ca 0 or -1
	mulhwu	r6, r7, r5		C carry = high(q*divisor)
	subf	r3, r3, r6		C carry + ca
	blr

EPILOGUE(mpn_modexact_1c_odd)
EPILOGUE(mpn_modexact_1_odd)
ASM_END()
