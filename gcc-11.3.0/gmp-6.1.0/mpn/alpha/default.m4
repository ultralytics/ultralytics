divert(-1)

dnl  m4 macros for alpha assembler (everywhere except unicos).


dnl  Copyright 2000, 2002-2004, 2013 Free Software Foundation, Inc.

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


dnl  Usage: ASM_START()
define(`ASM_START',
m4_assert_numargs(0)
`	.set noreorder
	.set noat')

dnl  Usage: X(value)
define(`X',
m4_assert_numargs(1)
`0x$1')

dnl  Usage: FLOAT64(label,value)
define(`FLOAT64',
m4_assert_numargs(2)
`	.align	3
$1:	.t_floating $2')


dnl  Called: PROLOGUE_cpu(GSYM_PREFIX`'foo[,gp|noalign])
dnl          EPILOGUE_cpu(GSYM_PREFIX`'foo)

define(`PROLOGUE_cpu',
m4_assert_numargs_range(1,2)
`ifelse(`$2',gp,,
`ifelse(`$2',noalign,,
`ifelse(`$2',,,`m4_error(`Unrecognised PROLOGUE parameter
')')')')dnl
	.text
ifelse(`$2',noalign,,`	ALIGN(16)')
	.globl	$1
	.ent	$1
$1:
	.frame r30,0,r26,0
ifelse(`$2',gp,`	ldgp	r29, 0(r27)
`$'$1..ng:')
	.prologue ifelse(`$2',gp,1,0)')

define(`EPILOGUE_cpu',
m4_assert_numargs(1)
`	.end	$1')


dnl  Usage: LDGP(dst,src)
dnl
dnl  Emit an "ldgp dst,src", but only if the system uses a GOT.

define(LDGP,
m4_assert_numargs(2)
`ldgp	`$1', `$2'')


dnl  Usage: EXTERN(variable_name)
define(`EXTERN',
m4_assert_numargs(1)
)

dnl  Usage: r0 ... r31
dnl         f0 ... f31
dnl
dnl  Map register names r0 to $0, and f0 to $f0, etc.
dnl  This is needed on all systems but Unicos
dnl
dnl  defreg() is used to protect the $ in $0 (otherwise it would represent a
dnl  macro argument).  Double quoting is used to protect the f0 in $f0
dnl  (otherwise it would be an infinite recursion).

forloop(i,0,31,`defreg(`r'i,$i)')
forloop(i,0,31,`deflit(`f'i,``$f''i)')


dnl  Usage: DATASTART(name,align)  or  DATASTART(name)
dnl         DATAEND()

define(`DATASTART',
m4_assert_numargs_range(1,2)
`	RODATA
	ALIGN(ifelse($#,1,2,$2))
$1:')
define(`DATAEND',
m4_assert_numargs(0)
)

dnl  Load a symbolic address into a register
define(`LEA',
m4_assert_numargs(2)
`lda	$1, $2')

dnl  Usage: ASM_END()
define(`ASM_END',
m4_assert_numargs(0)
)

divert
