divert(-1)

dnl  m4 macros for HPPA assembler.

dnl  Copyright 2002 Free Software Foundation, Inc.

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


dnl  hppa assembler comments are introduced with ";".
dnl
dnl  For cooperation with cpp, apparently lines "# 123" set the line number,
dnl  and other lines starting with a "#" are ignored.

changecom(;)


dnl  Called: PROLOGUE_cpu(GSYM_PREFIX`'foo)
dnl          EPILOGUE_cpu(GSYM_PREFIX`'foo)
dnl
dnl  These are the same as the basic PROLOGUE_cpu and EPILOGUE_cpu in
dnl  mpn/asm-defs.m4, but using .proc / .procend.  These are standard and on
dnl  an ELF system they do what .type and .size normally do.

define(`PROLOGUE_cpu',
m4_assert_numargs(1)
	`.code
	ALIGN(8)
	.export	`$1',entry
`$1'LABEL_SUFFIX'
	.proc
	.callinfo)	dnl  This is really bogus, but allows us to compile
			dnl  again on hppa machines.


define(`EPILOGUE_cpu',
m4_assert_numargs(1)
`	.procend')

divert
