divert(-1)

dnl  m4 macros for PowerPC assembler (32 and 64 bit).

dnl  Copyright 2000, 2002, 2003 Free Software Foundation, Inc.

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


dnl  Called: PROLOGUE_cpu(GSYM_PREFIX`'foo)
dnl
dnl  This is the same as the default in mpn/asm-defs.m4, but with ALIGN(4)
dnl  not 8.
dnl
dnl  4-byte alignment is normally enough, certainly it's what gcc gives.  We
dnl  don't want bigger alignment within PROLOGUE since it can introduce
dnl  padding into multiple-entrypoint routines, and with gas such padding is
dnl  zero words, which are not valid instructions.

define(`PROLOGUE_cpu',
m4_assert_numargs(1)
`	TEXT
	ALIGN(4)
	GLOBL	`$1' GLOBL_ATTR
	TYPE(`$1',`function')
`$1'LABEL_SUFFIX')


dnl  Usage: r0 ... r31, cr0 ... cr7
dnl
dnl  Registers names, either left as "r0" etc or mapped to plain 0 etc,
dnl  according to the result of the GMP_ASM_POWERPC_REGISTERS configure
dnl  test.

ifelse(WANT_R_REGISTERS,no,`
forloop(i,0,31,`deflit(`r'i,i)')
forloop(i,0,31,`deflit(`v'i,i)')
forloop(i,0,31,`deflit(`f'i,i)')
forloop(i,0,7, `deflit(`cr'i,i)')
')


dnl  Usage: ASSERT(cond,instructions)
dnl
dnl  If WANT_ASSERT is 1, output the given instructions and expect the given
dnl  flags condition to then be satisfied.  For example,
dnl
dnl         ASSERT(eq, `cmpwi r6, 123')
dnl
dnl  The instructions can be omitted to just assert a flags condition with
dnl  no extra calculation.  For example,
dnl
dnl         ASSERT(ne)
dnl
dnl  The condition can be omitted to just output the given instructions when
dnl  assertion checking is wanted.  For example,
dnl
dnl         ASSERT(, `mr r11, r0')
dnl
dnl  Using a zero word for an illegal instruction is probably not ideal,
dnl  since it marks the beginning of a traceback table in the 64-bit ABI.
dnl  But assertions are only for development, so it doesn't matter too much.

define(ASSERT,
m4_assert_numargs_range(1,2)
m4_assert_defined(`WANT_ASSERT')
`ifelse(WANT_ASSERT,1,
	`C ASSERT
	$2
ifelse(`$1',,,
`	b$1	L(ASSERT_ok`'ASSERT_counter)
	W32	0	C assertion failed
L(ASSERT_ok`'ASSERT_counter):
define(`ASSERT_counter',incr(ASSERT_counter))
')')')

define(ASSERT_counter,1)


divert
