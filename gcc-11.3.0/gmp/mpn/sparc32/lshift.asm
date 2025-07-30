dnl  SPARC mpn_lshift -- Shift a number left.

dnl  Copyright 1995, 1996, 2000 Free Software Foundation, Inc.

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

C INPUT PARAMETERS
C res_ptr	%o0
C src_ptr	%o1
C size		%o2
C cnt		%o3

ASM_START()
PROLOGUE(mpn_lshift)
	sll	%o2,2,%g1
	add	%o1,%g1,%o1	C make %o1 point at end of src
	ld	[%o1-4],%g2	C load first limb
	sub	%g0,%o3,%o5	C negate shift count
	add	%o0,%g1,%o0	C make %o0 point at end of res
	add	%o2,-1,%o2
	andcc	%o2,4-1,%g4	C number of limbs in first loop
	srl	%g2,%o5,%g1	C compute function result
	be	L(0)		C if multiple of 4 limbs, skip first loop
	st	%g1,[%sp+80]

	sub	%o2,%g4,%o2	C adjust count for main loop

L(loop0):
	ld	[%o1-8],%g3
	add	%o0,-4,%o0
	add	%o1,-4,%o1
	addcc	%g4,-1,%g4
	sll	%g2,%o3,%o4
	srl	%g3,%o5,%g1
	mov	%g3,%g2
	or	%o4,%g1,%o4
	bne	L(loop0)
	 st	%o4,[%o0+0]

L(0):	tst	%o2
	be	L(end)
	 nop

L(loop):
	ld	[%o1-8],%g3
	add	%o0,-16,%o0
	addcc	%o2,-4,%o2
	sll	%g2,%o3,%o4
	srl	%g3,%o5,%g1

	ld	[%o1-12],%g2
	sll	%g3,%o3,%g4
	or	%o4,%g1,%o4
	st	%o4,[%o0+12]
	srl	%g2,%o5,%g1

	ld	[%o1-16],%g3
	sll	%g2,%o3,%o4
	or	%g4,%g1,%g4
	st	%g4,[%o0+8]
	srl	%g3,%o5,%g1

	ld	[%o1-20],%g2
	sll	%g3,%o3,%g4
	or	%o4,%g1,%o4
	st	%o4,[%o0+4]
	srl	%g2,%o5,%g1

	add	%o1,-16,%o1
	or	%g4,%g1,%g4
	bne	L(loop)
	 st	%g4,[%o0+0]

L(end):	sll	%g2,%o3,%g2
	st	%g2,[%o0-4]
	retl
	ld	[%sp+80],%o0
EPILOGUE(mpn_lshift)
