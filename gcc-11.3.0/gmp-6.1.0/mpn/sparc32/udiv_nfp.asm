dnl  SPARC v7 __udiv_qrnnd division support, used from longlong.h.
dnl  This is for v7 CPUs without a floating-point unit.

dnl  Copyright 1993, 1994, 1996, 2000 Free Software Foundation, Inc.

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
C rem_ptr	o0
C n1		o1
C n0		o2
C d		o3

ASM_START()
PROLOGUE(mpn_udiv_qrnnd)
	tst	%o3
	bneg	L(largedivisor)
	mov	8,%g1

	b	L(p1)
	addxcc	%o2,%o2,%o2

L(plop):
	bcc	L(n1)
	addxcc	%o2,%o2,%o2
L(p1):	addx	%o1,%o1,%o1
	subcc	%o1,%o3,%o4
	bcc	L(n2)
	addxcc	%o2,%o2,%o2
L(p2):	addx	%o1,%o1,%o1
	subcc	%o1,%o3,%o4
	bcc	L(n3)
	addxcc	%o2,%o2,%o2
L(p3):	addx	%o1,%o1,%o1
	subcc	%o1,%o3,%o4
	bcc	L(n4)
	addxcc	%o2,%o2,%o2
L(p4):	addx	%o1,%o1,%o1
	addcc	%g1,-1,%g1
	bne	L(plop)
	subcc	%o1,%o3,%o4
	bcc	L(n5)
	addxcc	%o2,%o2,%o2
L(p5):	st	%o1,[%o0]
	retl
	xnor	%g0,%o2,%o0

L(nlop):
	bcc	L(p1)
	addxcc	%o2,%o2,%o2
L(n1):	addx	%o4,%o4,%o4
	subcc	%o4,%o3,%o1
	bcc	L(p2)
	addxcc	%o2,%o2,%o2
L(n2):	addx	%o4,%o4,%o4
	subcc	%o4,%o3,%o1
	bcc	L(p3)
	addxcc	%o2,%o2,%o2
L(n3):	addx	%o4,%o4,%o4
	subcc	%o4,%o3,%o1
	bcc	L(p4)
	addxcc	%o2,%o2,%o2
L(n4):	addx	%o4,%o4,%o4
	addcc	%g1,-1,%g1
	bne	L(nlop)
	subcc	%o4,%o3,%o1
	bcc	L(p5)
	addxcc	%o2,%o2,%o2
L(n5):	st	%o4,[%o0]
	retl
	xnor	%g0,%o2,%o0

L(largedivisor):
	and	%o2,1,%o5	C %o5 = n0 & 1

	srl	%o2,1,%o2
	sll	%o1,31,%g2
	or	%g2,%o2,%o2	C %o2 = lo(n1n0 >> 1)
	srl	%o1,1,%o1	C %o1 = hi(n1n0 >> 1)

	and	%o3,1,%g2
	srl	%o3,1,%g3	C %g3 = floor(d / 2)
	add	%g3,%g2,%g3	C %g3 = ceil(d / 2)

	b	L(Lp1)
	addxcc	%o2,%o2,%o2

L(Lplop):
	bcc	L(Ln1)
	addxcc	%o2,%o2,%o2
L(Lp1):	addx	%o1,%o1,%o1
	subcc	%o1,%g3,%o4
	bcc	L(Ln2)
	addxcc	%o2,%o2,%o2
L(Lp2):	addx	%o1,%o1,%o1
	subcc	%o1,%g3,%o4
	bcc	L(Ln3)
	addxcc	%o2,%o2,%o2
L(Lp3):	addx	%o1,%o1,%o1
	subcc	%o1,%g3,%o4
	bcc	L(Ln4)
	addxcc	%o2,%o2,%o2
L(Lp4):	addx	%o1,%o1,%o1
	addcc	%g1,-1,%g1
	bne	L(Lplop)
	subcc	%o1,%g3,%o4
	bcc	L(Ln5)
	addxcc	%o2,%o2,%o2
L(Lp5):	add	%o1,%o1,%o1	C << 1
	tst	%g2
	bne	L(oddp)
	add	%o5,%o1,%o1
	st	%o1,[%o0]
	retl
	xnor	%g0,%o2,%o0

L(Lnlop):
	bcc	L(Lp1)
	addxcc	%o2,%o2,%o2
L(Ln1):	addx	%o4,%o4,%o4
	subcc	%o4,%g3,%o1
	bcc	L(Lp2)
	addxcc	%o2,%o2,%o2
L(Ln2):	addx	%o4,%o4,%o4
	subcc	%o4,%g3,%o1
	bcc	L(Lp3)
	addxcc	%o2,%o2,%o2
L(Ln3):	addx	%o4,%o4,%o4
	subcc	%o4,%g3,%o1
	bcc	L(Lp4)
	addxcc	%o2,%o2,%o2
L(Ln4):	addx	%o4,%o4,%o4
	addcc	%g1,-1,%g1
	bne	L(Lnlop)
	subcc	%o4,%g3,%o1
	bcc	L(Lp5)
	addxcc	%o2,%o2,%o2
L(Ln5):	add	%o4,%o4,%o4	C << 1
	tst	%g2
	bne	L(oddn)
	add	%o5,%o4,%o4
	st	%o4,[%o0]
	retl
	xnor	%g0,%o2,%o0

L(oddp):
	xnor	%g0,%o2,%o2
	C q' in %o2. r' in %o1
	addcc	%o1,%o2,%o1
	bcc	L(Lp6)
	addx	%o2,0,%o2
	sub	%o1,%o3,%o1
L(Lp6):	subcc	%o1,%o3,%g0
	bcs	L(Lp7)
	subx	%o2,-1,%o2
	sub	%o1,%o3,%o1
L(Lp7):	st	%o1,[%o0]
	retl
	mov	%o2,%o0

L(oddn):
	xnor	%g0,%o2,%o2
	C q' in %o2. r' in %o4
	addcc	%o4,%o2,%o4
	bcc	L(Ln6)
	addx	%o2,0,%o2
	sub	%o4,%o3,%o4
L(Ln6):	subcc	%o4,%o3,%g0
	bcs	L(Ln7)
	subx	%o2,-1,%o2
	sub	%o4,%o3,%o4
L(Ln7):	st	%o4,[%o0]
	retl
	mov	%o2,%o0
EPILOGUE(mpn_udiv_qrnnd)
