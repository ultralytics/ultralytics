dnl  PowerPC 750 mpn_com -- mpn bitwise one's complement

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


C                cycles/limb
C 603e:            ?
C 604e:            3.0
C 75x (G3):        2.0
C 7400,7410 (G4):  2.0
C 744x,745x (G4+): 3.0

C void mpn_com (mp_ptr dst, mp_srcptr src, mp_size_t size);
C
C This loop form is necessary for the claimed speed.

ASM_START()
PROLOGUE(mpn_com)

	C r3	dst
	C r4	src
	C r5	size

	mtctr	r5		C size
	lwz	r5, 0(r4)	C src low limb

	sub	r4, r4, r3	C src-dst
	subi	r3, r3, 4	C dst-4

	addi	r4, r4, 8	C src-dst+8
	bdz	L(one)

L(top):
	C r3	&dst[i-1]
	C r4	src-dst
	C r5	src[i]
	C r6	scratch

	not	r6, r5		C ~src[i]
	lwzx	r5, r4,r3	C src[i+1]

	stwu	r6, 4(r3)	C dst[i]
	bdnz	L(top)

L(one):
	not	r6, r5

	stw	r6, 4(r3)	C dst[size-1]
	blr

EPILOGUE()
