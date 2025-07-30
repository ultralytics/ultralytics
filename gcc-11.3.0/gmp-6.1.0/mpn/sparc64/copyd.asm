dnl  SPARC v9 mpn_copyd -- Copy a limb vector, decrementing.

dnl  Copyright 1999-2003 Free Software Foundation, Inc.

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

C		   cycles/limb
C UltraSPARC 1&2:	 2
C UltraSPARC 3:		 2.5
C UltraSPARC T1:	17
C UltraSPARC T3:	 6
C UltraSPARC T4/T5:	 2

C INPUT PARAMETERS
C rptr	%o0
C sptr	%o1
C n	%o2

ASM_START()
	REGISTER(%g2,#scratch)
	REGISTER(%g3,#scratch)
PROLOGUE(mpn_copyd)
	sllx	%o2,3,%g1
	add	%g1,%o0,%o0
	add	%g1,%o1,%o1
	addcc	%o2,-8,%o2
	bl,pt	%xcc,L(end01234567)
	nop
L(loop1):
	ldx	[%o1-8],%g1
	ldx	[%o1-16],%g2
	ldx	[%o1-24],%g3
	ldx	[%o1-32],%g4
	ldx	[%o1-40],%g5
	ldx	[%o1-48],%o3
	ldx	[%o1-56],%o4
	ldx	[%o1-64],%o5
	add	%o1,-64,%o1
	stx	%g1,[%o0-8]
	stx	%g2,[%o0-16]
	stx	%g3,[%o0-24]
	stx	%g4,[%o0-32]
	stx	%g5,[%o0-40]
	stx	%o3,[%o0-48]
	stx	%o4,[%o0-56]
	stx	%o5,[%o0-64]
	addcc	%o2,-8,%o2
	bge,pt	%xcc,L(loop1)
	add	%o0,-64,%o0
L(end01234567):
	addcc	%o2,8,%o2
	bz,pn	%xcc,L(end)
	nop
L(loop2):
	ldx	[%o1-8],%g1
	add	%o1,-8,%o1
	addcc	%o2,-1,%o2
	stx	%g1,[%o0-8]
	bg,pt	%xcc,L(loop2)
	add	%o0,-8,%o0
L(end):	retl
	nop
EPILOGUE(mpn_copyd)
