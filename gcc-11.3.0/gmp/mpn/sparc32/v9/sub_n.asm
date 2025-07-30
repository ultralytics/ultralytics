dnl  SPARC mpn_sub_n -- Subtract two limb vectors of the same length > 0 and
dnl  store difference in a third limb vector.

dnl  Copyright 2001 Free Software Foundation, Inc.

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
define(rp,%o0)
define(s1p,%o1)
define(s2p,%o2)
define(n,%o3)
define(cy,%g1)

C This code uses 64-bit operations on `o' and `g' registers.  It doesn't
C require that `o' registers' upper 32 bits are preserved by the operating
C system, but if they are not, they must be zeroed.  That is indeed what
C happens at least on Slowaris 2.5 and 2.6.

C On UltraSPARC 1 and 2, this code runs at 3 cycles/limb from the Dcache and at
C about 10 cycles/limb from the Ecache.

ASM_START()
PROLOGUE(mpn_sub_n)
	lduw	[s1p+0],%o4
	lduw	[s2p+0],%o5
	addcc	n,-2,n
	bl,pn	%icc,L(end1)
	lduw	[s1p+4],%g2
	lduw	[s2p+4],%g3
	be,pn	%icc,L(end2)
	mov	0,cy

	.align	16
L(loop):
	sub	%o4,%o5,%g4
	add	rp,8,rp
	lduw	[s1p+8],%o4
	fitod	%f0,%f2
C ---
	sub	%g4,cy,%g4
	addcc	n,-1,n
	lduw	[s2p+8],%o5
	fitod	%f0,%f2
C ---
	srlx	%g4,63,cy
	add	s2p,8,s2p
	stw	%g4,[rp-8]
	be,pn	%icc,L(exito)+4
C ---
	sub	%g2,%g3,%g4
	addcc	n,-1,n
	lduw	[s1p+12],%g2
	fitod	%f0,%f2
C ---
	sub	%g4,cy,%g4
	add	s1p,8,s1p
	lduw	[s2p+4],%g3
	fitod	%f0,%f2
C ---
	srlx	%g4,63,cy
	bne,pt	%icc,L(loop)
	stw	%g4,[rp-4]
C ---
L(exite):
	sub	%o4,%o5,%g4
	sub	%g4,cy,%g4
	srlx	%g4,63,cy
	stw	%g4,[rp+0]
	sub	%g2,%g3,%g4
	sub	%g4,cy,%g4
	stw	%g4,[rp+4]
	retl
	srlx	%g4,63,%o0

L(exito):
	sub	%g2,%g3,%g4
	sub	%g4,cy,%g4
	srlx	%g4,63,cy
	stw	%g4,[rp-4]
	sub	%o4,%o5,%g4
	sub	%g4,cy,%g4
	stw	%g4,[rp+0]
	retl
	srlx	%g4,63,%o0

L(end1):
	sub	%o4,%o5,%g4
	stw	%g4,[rp+0]
	retl
	srlx	%g4,63,%o0

L(end2):
	sub	%o4,%o5,%g4
	srlx	%g4,63,cy
	stw	%g4,[rp+0]
	sub	%g2,%g3,%g4
	sub	%g4,cy,%g4
	stw	%g4,[rp+4]
	retl
	srlx	%g4,63,%o0
EPILOGUE(mpn_sub_n)
