dnl  SPARC v9 64-bit mpn_submul_1 -- Multiply a limb vector with a limb and
dnl  subtract the result from a second limb vector.

dnl  Copyright 2001-2003 Free Software Foundation, Inc.

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
C UltraSPARC 1&2:     18
C UltraSPARC 3:	      23

C INPUT PARAMETERS
C rp	i0
C up	i1
C n	i2
C v	i3

ASM_START()
	REGISTER(%g2,#scratch)

PROLOGUE(mpn_submul_1)
	save	%sp,-176,%sp

	sllx	%i2, 3, %g2
	or	%g0, %i1, %o1
	add	%g2, 15, %o0
	or	%g0, %i2, %o2
	and	%o0, -16, %o0
	sub	%sp, %o0, %sp
	add	%sp, 2223, %o0
	or	%g0, %o0, %l0
	call	mpn_mul_1
	or	%g0, %i3, %o3
	or	%g0, %o0, %l1		C preserve carry value from mpn_mul_1
	or	%g0, %i0, %o0
	or	%g0, %i0, %o1
	or	%g0, %l0, %o2
	call	mpn_sub_n
	or	%g0, %i2, %o3
	ret
	restore	%l1, %o0, %o0		C sum carry values
EPILOGUE(mpn_submul_1)
