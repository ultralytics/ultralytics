dnl  SPARC T3/T4/T5 mpn_invert_limb.

dnl  Contributed to the GNU project by Torbjörn Granlund.

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

C                  cycles/limb
C UltraSPARC T3:	 ?
C UltraSPARC T4/T5:	 ?

C INPUT PARAMETERS
define(`d',  `%o0')

ASM_START()
	REGISTER(%g2,#scratch)
	REGISTER(%g3,#scratch)
PROLOGUE(mpn_invert_limb)
	srlx	d, 54, %g1
	LEA64(approx_tab, g2, g3)
	and	%g1, 0x1fe, %g1
	srlx	d, 24, %g4
	lduh	[%g2+%g1], %g3
	add	%g4, 1, %g4
	sllx	%g3, 11, %g2
	add	%g2, -1, %g2
	mulx	%g3, %g3, %g3
	mulx	%g3, %g4, %g3
	srlx	%g3, 40, %g3
	sub	%g2, %g3, %g2
	sllx	%g2, 60, %g1
	mulx	%g2, %g2, %g3
	mulx	%g3, %g4, %g4
	sub	%g1, %g4, %g1
	srlx	%g1, 47, %g1
	sllx	%g2, 13, %g2
	add	%g1, %g2, %g1
	and	d, 1, %g2
	srlx	%g1, 1, %g4
	sub	%g0, %g2, %g3
	and	%g4, %g3, %g3
	srlx	d, 1, %g4
	add	%g4, %g2, %g2
	mulx	%g1, %g2, %g2
	sub	%g3, %g2, %g2
	umulxhi(%g1, %g2, %g2)
	srlx	%g2, 1, %g2
	sllx	%g1, 31, %g1
	add	%g2, %g1, %g1
	mulx	%g1, d, %g3
	umulxhi(d, %g1, %g4)
	addcc	%g3, d, %g0
	addxc(	%g4, d, %o0)
	jmp	%o7+8
	 sub	%g1, %o0, %o0
EPILOGUE()

	RODATA
	ALIGN(2)
	TYPE(	approx_tab, object)
	SIZE(	approx_tab, 512)
approx_tab:
forloop(i,256,512-1,dnl
`	.half	eval(0x7fd00/i)
')dnl
