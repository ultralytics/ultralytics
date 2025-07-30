dnl  Table used for mpn_invert_limb

dnl  Contributed to the GNU project by Torbjorn Granlund and Niels Möller.

dnl  Copyright 2004, 2007-2009, 2011-2013 Free Software Foundation, Inc.

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

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

PROTECT(`mpn_invert_limb_table')

ASM_START()
C Table entry X contains floor (0x7fd00 / (0x100 + X))

	RODATA
	ALIGN(2)
	GLOBL mpn_invert_limb_table
mpn_invert_limb_table:
forloop(i,256,512-1,dnl
`	.value	eval(0x7fd00/i)
')dnl
ASM_END()
