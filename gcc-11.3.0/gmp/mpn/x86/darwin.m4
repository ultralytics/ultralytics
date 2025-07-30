divert(-1)
dnl  Copyright 2007, 2011, 2012, 2014 Free Software Foundation, Inc.

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

define(`DARWIN')


dnl  Usage LEA(symbol,reg)
dnl  Usage LEAL(symbol_local_to_file,reg)
dnl
dnl  We maintain lists of stuff to append in load_eip and darwin_bd.  The
dnl  `index' stuff is needed to suppress repeated definitions.  To avoid
dnl  getting fooled by "var" and "var1", we add 'bol ' (the end of
dnl  'indirect_symbol') at the beginning and and a newline at the end.  This
dnl  might be a bit fragile.

define(`LEA',
m4_assert_numargs(2)
`ifdef(`PIC',`
ifelse(index(defn(`load_eip'), `$2'),-1,
`m4append(`load_eip',
`	TEXT
	ALIGN(16)
L(movl_eip_`'substr($2,1)):
	movl	(%esp), $2
	ret_internal
')')
ifelse(index(defn(`darwin_bd'), `bol $1
'),-1,
`m4append(`darwin_bd',
`	.section __IMPORT,__pointers,non_lazy_symbol_pointers
L($1`'$non_lazy_ptr):
	.indirect_symbol $1
	.long	 0
')')
	call	L(movl_eip_`'substr($2,1))
	movl	L($1`'$non_lazy_ptr)-.($2), $2
',`
	movl	`$'$1, $2
')')

define(`LEAL',
m4_assert_numargs(2)
`ifdef(`PIC',`
ifelse(index(defn(`load_eip'), `$2'),-1,
`m4append(`load_eip',
`	TEXT
	ALIGN(16)
L(movl_eip_`'substr($2,1)):
	movl	(%esp), $2
	ret_internal
')')
	call	L(movl_eip_`'substr($2,1))
	leal	$1-.($2), $2
',`
	movl	`$'$1, $2
')')


dnl ASM_END

define(`ASM_END',`load_eip`'darwin_bd')

define(`load_eip', `')		dnl updated in LEA
define(`darwin_bd', `')		dnl updated in LEA


dnl  Usage: CALL(funcname)
dnl

define(`CALL',
m4_assert_numargs(1)
`call	GSYM_PREFIX`'$1')

undefine(`PIC_WITH_EBX')

divert`'dnl
