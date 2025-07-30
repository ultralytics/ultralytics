divert(-1)
dnl  m4 macros for powerpc32 GNU/Linux assembly.

dnl  Copyright 2003, 2005, 2006 Free Software Foundation, Inc.

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

define(`ASM_START',`')

dnl  Called: PROLOGUE_cpu(GSYM_PREFIX`'foo[,toc])
dnl          EPILOGUE_cpu(GSYM_PREFIX`'foo)
dnl

define(`PROLOGUE_cpu',
m4_assert_numargs_range(1,2)
`ifelse(`$2',toc,,
`ifelse(`$2',,,`m4_error(`Unrecognised PROLOGUE parameter')')')dnl
	.section	".text"
	.align	3
	.globl	$1
	.type	$1, @function
$1:')

define(`EPILOGUE_cpu',
m4_assert_numargs(1)
`	.size	$1, .-$1')

define(`LEA',
m4_assert_numargs(2)
`ifdef(`PIC',`
	mflr	r0
	bcl	20, 31, 1f
1:	mflr	$1
	addis	$1, $1, (_GLOBAL_OFFSET_TABLE_-1b)@ha
	addi	$1, $1, (_GLOBAL_OFFSET_TABLE_-1b)@l
	mtlr	r0
	lwz	$1, $2@got($1)
',`
	lis	$1, $2@ha
	la	$1, $2@l($1)
')')


define(`LEAL',
m4_assert_numargs(2)
`LEA($1,$2)')


define(`EXTERN',
m4_assert_numargs(1)
`dnl')

define(`DEF_OBJECT',
m4_assert_numargs_range(1,2)
`
	.section	.rodata
	ALIGN(ifelse($#,1,2,$2))
	.type	$1, @object
$1:
')

define(`END_OBJECT',
m4_assert_numargs(1)
`	.size	$1, .-$1')

define(`ASM_END', `dnl')

ifdef(`PIC',`
define(`PIC_SLOW')')

dnl  64-bit "long long" parameters are put in an even-odd pair, skipping an
dnl  even register if that was in turn.  I wish somebody could explain why that
dnl  is a good idea.
define(`BROKEN_LONGLONG_PARAM')

divert
