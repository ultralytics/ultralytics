dnl  PowerPC 750 mpn_lshift -- mpn left shift.

dnl  Copyright 2002, 2003 Free Software Foundation, Inc.

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


C       cycles/limb
C 750:     3.0
C 7400:    3.0


C mp_limb_t mpn_lshift (mp_ptr dst, mp_srcptr src, mp_size_t size,
C                       unsigned shift);
C
C This code is the same per-limb speed as mpn/powerpc32/lshift.asm, but
C smaller and saving about 30 or so cycles of overhead.

ASM_START()
PROLOGUE(mpn_lshift)

	C r3	dst
	C r4	src
	C r5	size
	C r6	shift

	mtctr	r5		C size
	slwi	r5, r5, 2	C 4*size

	subfic	r7, r6, 32	C 32-shift
	add	r4, r4, r5	C &src[size]

	add	r5, r3, r5	C &dst[size]
	lwz	r8, -4(r4)	C src[size-1]
	bdz	L(one)

	lwzu	r9, -8(r4)	C src[size-2]

	srw	r3, r8, r7	C return value
	slw	r8, r8, r6	C src[size-1] << shift
	bdz	L(two)


L(top):
	C r3	return value
	C r4	src, incrementing
	C r5	dst, incrementing
	C r6	lshift
	C r7	32-shift
	C r8	src[i+1] << shift
	C r9	src[i]
	C r10

	lwzu	r10, -4(r4)
	srw	r11, r9, r7

	or	r8, r8, r11
	stwu	r8, -4(r5)

	slw	r8, r9, r6
	bdz	L(odd)

	C r8	src[i+1] << shift
	C r9
	C r10	src[i]

	lwzu	r9, -4(r4)
	srw	r11, r10, r7

	or	r8, r8, r11
	stwu	r8, -4(r5)

	slw	r8, r10, r6
	bdnz	L(top)


L(two):
	C r3	return value
	C r4
	C r5	&dst[2]
	C r6	shift
	C r7	32-shift
	C r8	src[1] << shift
	C r9	src[0]
	C r10

	srw	r11, r9, r7
	slw	r12, r9, r6	C src[0] << shift

	or	r8, r8, r11
	stw	r12, -8(r5)	C dst[0]

	stw	r8, -4(r5)	C dst[1]
	blr


L(odd):
	C r3	return value
	C r4
	C r5	&dst[2]
	C r6	shift
	C r7	32-shift
	C r8	src[1] << shift
	C r9
	C r10	src[0]

	srw	r11, r10, r7
	slw	r12, r10, r6

	or	r8, r8, r11
	stw	r12, -8(r5)	C dst[0]

	stw	r8, -4(r5)	C dst[1]
	blr


L(one):
	C r5	&dst[1]
	C r6	shift
	C r7	32-shift
	C r8	src[0]

	srw	r3, r8, r7	C return value
	slw	r8, r8, r6	C src[size-1] << shift

	stw	r8, -4(r5)	C dst[0]
	blr

EPILOGUE(mpn_lshift)
