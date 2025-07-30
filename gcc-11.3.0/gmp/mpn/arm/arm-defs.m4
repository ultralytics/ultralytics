divert(-1)

dnl  m4 macros for ARM assembler.

dnl  Copyright 2001, 2012, 2013 Free Software Foundation, Inc.

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


dnl  Standard commenting is with @, the default m4 # is for constants and we
dnl  don't want to disable macro expansions in or after them.

changecom(@&*$)


dnl  APCS register names.

deflit(a1,r0)
deflit(a2,r1)
deflit(a3,r2)
deflit(a4,r3)
deflit(v1,r4)
deflit(v2,r5)
deflit(v3,r6)
deflit(v4,r7)
deflit(v5,r8)
deflit(v6,r9)
deflit(sb,r9)
deflit(v7,r10)
deflit(sl,r10)
deflit(fp,r11)
deflit(ip,r12)
deflit(sp,r13)
deflit(lr,r14)
deflit(pc,r15)


define(`lea_list', `')
define(`lea_num',0)

dnl  LEA(reg,gmp_symbol)
dnl
dnl  Load the address of gmp_symbol into a register.  The gmp_symbol must be
dnl  either local or protected/hidden, since we assume it has a fixed distance
dnl  from the point of use.

define(`LEA',`dnl
ldr	$1, L(ptr`'lea_num)
ifdef(`PIC',dnl
`dnl
L(bas`'lea_num):dnl
	add	$1, $1, pc`'dnl
	m4append(`lea_list',`
L(ptr'lea_num`):	.word	GSYM_PREFIX`'$2-L(bas'lea_num`)-8')
	define(`lea_num', eval(lea_num+1))dnl
',`dnl
	m4append(`lea_list',`
L(ptr'lea_num`):	.word	GSYM_PREFIX`'$2')
	define(`lea_num', eval(lea_num+1))dnl
')dnl
')

define(`EPILOGUE_cpu',
`lea_list
	SIZE(`$1',.-`$1')')

divert
