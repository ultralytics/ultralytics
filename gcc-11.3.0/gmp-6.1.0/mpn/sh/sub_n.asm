dnl  SH mpn_sub_n -- Subtract two limb vectors of the same length > 0 and store
dnl  difference in a third limb vector.

dnl  Copyright 1995, 1997, 2000, 2011 Free Software Foundation, Inc.

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
C rp		r4
C up		r5
C vp		r6
C n		r7

changecom(blah)			C disable # to make all C comments below work

ASM_START()
PROLOGUE(mpn_sub_n)
	mov	#0,r3		C clear cy save reg

L(top):	mov.l	@r5+,r1
	mov.l	@r6+,r2
	shlr	r3		C restore cy
	subc	r2,r1
	movt	r3		C save cy
	mov.l	r1,@r4
	dt	r7
	bf.s	L(top)
	 add	#4,r4

	rts
	mov	r3,r0		C return carry-out from most significant limb
EPILOGUE()
