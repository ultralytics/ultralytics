dnl  SPARC v9 mpn_popcount for T3/T4.

dnl  Contributed to the GNU project by David Miller.

dnl  Copyright 2013 Free Software Foundation, Inc.

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
C UltraSPARC T3:	15
C UltraSPARC T4:	 2.5

C INPUT PARAMETERS
define(`up',   `%o0')
define(`n',    `%o1')
define(`pcnt', `%o5')

ASM_START()
	REGISTER(%g2,#scratch)
	REGISTER(%g3,#scratch)
PROLOGUE(mpn_popcount)
	subcc	n, 1, n
	be	L(final_one)
	 clr	pcnt
L(top):
	ldx	[up + 0], %g1
	sub	n, 2, n
	ldx	[up + 8], %o4
	add	up, 16, up
	popc	%g1, %g2
	popc	%o4, %g3
	add	pcnt, %g2, pcnt
	brgz	n, L(top)
	 add	pcnt, %g3, pcnt
	brlz,pt	n, L(done)
	 nop
L(final_one):
	ldx	[up + 0], %g1
	popc	%g1, %g2
	add	pcnt, %g2, pcnt
L(done):
	retl
	 mov	pcnt, %o0
EPILOGUE()
