dnl  PowerPC-64 mpn_invert_limb -- Invert a normalized limb.

dnl  Copyright 2015 Free Software Foundation, Inc.

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

C                  cycles/limb (approximate)
C POWER3/PPC630         -
C POWER4/PPC970         -
C POWER5                -
C POWER6                -
C POWER7                ?
C POWER8               32

C This runs on POWER7 and later, but is faster only on later CPUs.
C We might want to inline this, considering its small footprint.

ASM_START()
PROLOGUE(mpn_invert_limb)
	sldi.	r4, r3, 1
	neg	r5, r3
	divdeu	r3, r5, r3
	beq-	L(1)
	blr
L(1):	li	r3, -1
	blr
EPILOGUE()
