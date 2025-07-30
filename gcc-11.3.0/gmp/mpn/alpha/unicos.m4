divert(-1)

dnl  m4 macros for alpha assembler on unicos.


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


dnl  Note that none of the standard GMP_ASM_ autoconf tests are done for
dnl  unicos, so none of the config.m4 results can be used here.

dnl  No underscores on unicos
define(`GSYM_PREFIX')

define(`ASM_START',
m4_assert_numargs(0)
`	.ident	dummy')

define(`X',
m4_assert_numargs(1)
`^X$1')

define(`FLOAT64',
m4_assert_numargs(2)
`	.psect	$1@crud,data
$1:	.t_floating $2
	.endp')

dnl  Called: PROLOGUE_cpu(GSYM_PREFIX`'foo[,gp|noalign])
dnl          EPILOGUE_cpu(GSYM_PREFIX`'foo)

define(`PROLOGUE_cpu',
m4_assert_numargs_range(1,2)
`ifelse(`$2',gp,,
`ifelse(`$2',noalign,,
`ifelse(`$2',,,`m4_error(`Unrecognised PROLOGUE parameter
')')')')dnl
	.stack	192		; What does this mean?  Only Cray knows.
	.psect	$1@code,code,cache
$1::')

define(`EPILOGUE_cpu',
m4_assert_numargs(1)
`	.endp')


dnl  Usage: LDGP(dst,src)
dnl
dnl  Emit an "ldgp dst,src", but only on systems using a GOT (which unicos
dnl  doesn't).

define(LDGP,
m4_assert_numargs(2)
)


dnl  Usage: EXTERN(variable_name)
define(`EXTERN',
m4_assert_numargs(1)
`	.extern	$1')

define(`DATASTART',
m4_assert_numargs_range(1,2)
`	.psect	$1@crud,data
	ALIGN(ifelse($#,1,2,$2))
$1:')

define(`DATAEND',
m4_assert_numargs(0)
`	.endp')

define(`ASM_END',
m4_assert_numargs(0)
`	.end')

define(`cvttqc',
m4_assert_numargs(-1)
`cvttq/c')

dnl  Load a symbolic address into a register
define(`LEA',
m4_assert_numargs(2)
	`laum	$1,  $2(r31)
	sll	$1,  32,   $1
	lalm	$1,  $2($1)
	lal	$1,  $2($1)')


dnl  Usage: ALIGN(bytes)
dnl
dnl  Unicos assembler .align emits zeros, even in code segments, so disable
dnl  aligning.
dnl
dnl  GCC uses a macro emiting nops until the desired alignment is reached
dnl  (see unicosmk_file_start in alpha.c).  Could do something like that if
dnl  we cared.  The maximum desired alignment must be established at the
dnl  start of the section though, since of course emitting nops only
dnl  advances relative to the section beginning.

define(`ALIGN',
m4_assert_numargs(1)
)


divert
