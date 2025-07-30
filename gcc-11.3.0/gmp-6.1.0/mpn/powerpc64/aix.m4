divert(-1)
dnl  m4 macros for AIX 64-bit assembly.

dnl  Copyright 2000-2002, 2005, 2006, 2010, 2012 Free Software Foundation, Inc.

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

define(`ASM_START',
	`.machine	"any"
	.toc')

dnl  Called: PROLOGUE_cpu(GSYM_PREFIX`'foo[,toc])
dnl          EPILOGUE_cpu(GSYM_PREFIX`'foo)
dnl
dnl  Don't want ELF style .size in the epilogue.

define(`PROLOGUE_cpu',
m4_assert_numargs_range(1,2)
`ifelse(`$2',toc,,
`ifelse(`$2',,,`m4_error(`Unrecognised PROLOGUE parameter')')')dnl
	.globl	$1
	.globl	.$1
	.csect	[DS], 3
$1:
	.llong	.$1, TOC[tc0], 0
	.csect	.$1[PR], 6
.$1:')

define(`EPILOGUE_cpu',
m4_assert_numargs(1)
`')

define(`TOC_ENTRY', `')

define(`LEA',
m4_assert_numargs(2)
`define(`TOC_ENTRY',
`	.toc
..$2:	.tc	$2[TC], $2')'
	`ld	$1, ..$2(2)')

define(`LEAL',
m4_assert_numargs(2)
`LEA($1,$2)')


define(`EXTERN',
m4_assert_numargs(1)
`	.globl	$1')

define(`EXTERN_FUNC',
m4_assert_numargs(1)
`	.globl	.$1')

define(`DEF_OBJECT',
m4_assert_numargs_range(1,2)
`	.csect	[RO], 3
	ALIGN(ifelse($#,1,2,$2))
$1:
')

define(`END_OBJECT',
m4_assert_numargs(1))

define(`CALL',
	`bl	.$1
	nop')

define(`ASM_END', `TOC_ENTRY')

undefine(`EXTRA_REGISTER')

divert
