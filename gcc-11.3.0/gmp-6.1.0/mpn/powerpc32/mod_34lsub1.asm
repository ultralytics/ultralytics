dnl  PowerPC-32 mpn_mod_34lsub1 -- mpn remainder mod 2^24-1.

dnl  Copyright 2002, 2003, 2005 Free Software Foundation, Inc.

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
C 603e:            ?
C 604e:            3
C 75x (G3):        3
C 7400,7410 (G4):  3
C 744x,745x (G4+): 3
C power4/ppc970:   2.5
C power5:          2.5

C mp_limb_t mpn_mod_34lsub1 (mp_srcptr src, mp_size_t size)
C
C There seems no need to schedule the loads back, the code is still 3.0 c/l
C on 750/7400 no matter where they're placed.
C
C Alternatives:
C
C Fetching half words would allow add instead for accumulating, instead of
C adde and its serialization.  An outer loop would be required though, since
C 2^16 halfwords can overflow.  lhz+add would be 2.0 c/l, but if there's
C also a bdz or bdnz for each and a pointer update say every three limbs
C then the total would be 2.67 c/l which isn't much faster than the current
C simpler code.

ASM_START()
PROLOGUE(mpn_mod_34lsub1)

	C r3	src
	C r4	size

	mtctr	r4
	addic	r6, r3, 8		C &src[2], and clear CA

	lwz	r3, 0(r3)		C acc0 = src[0]
	bdz	L(done)

	lwz	r4, -4(r6)		C acc1 = src[1]
	bdz	L(two)

	lwz	r5, 0(r6)		C acc2 = src[2]
	lis	r7, 0			C no carry if just three limbs

	bdz	L(three)
	lis	r7, 1			C 0x10000 carry pos

L(top):
	C r3	acc0
	C r4	acc1
	C r5	acc2
	C r6	src, incrementing
	C r7	carry pos

	lwz	r0, 4(r6)
	adde	r3, r3, r0
	bdz	L(end0)

	lwz	r0, 8(r6)
	adde	r4, r4, r0
	bdz	L(end1)

	lwzu	r0, 12(r6)
	adde	r5, r5, r0
	bdnz	L(top)


	srwi	r7, r7, 8
L(end0):
	srwi	r7, r7, 8
L(end1):
	subfe	r0, r0, r0		C -1 if not CA

	andc	r7, r7, r0		C final carry, 0x10000, 0x100, 1 or 0
L(three):
	rlwinm	r6, r3, 0,8,31		C acc0 low

	add	r7, r7, r6
	rlwinm	r6, r3, 8,24,31		C acc0 high

	add	r7, r7, r6
	rlwinm	r6, r4, 8,8,23		C acc1 low

	add	r7, r7, r6
	rlwinm	r6, r4, 16,16,31	C acc1 high

	add	r7, r7, r6
	rlwinm	r6, r5, 16,8,15		C acc2 low

	add	r7, r7, r6
	rlwinm	r6, r5, 24,8,31		C acc2 high

	add	r3, r7, r6

L(done):
	blr

L(two):
	C r3	acc0
	C r4	acc1

	rlwinm	r5, r3, 8,24,31		C acc0 high
	rlwinm	r3, r3, 0,8,31		C acc0 low

	add	r3, r3, r5		C acc0 high + low
	rlwinm	r5, r4, 16,16,31	C acc1 high

	add	r3, r3, r5		C add acc1 high
	rlwinm	r5, r4, 8,8,23		C acc1 low

	add	r3, r3, r5		C add acc1 low

	blr

EPILOGUE()
