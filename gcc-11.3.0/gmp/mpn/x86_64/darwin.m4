divert(-1)
dnl  Copyright 2008, 2011, 2012 Free Software Foundation, Inc.
dnl
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

define(`LEA',`dnl
ifdef(`PIC',
	`lea	$1(%rip), $2'
,
	`movabs	`$'$1, $2')
')

dnl  Usage: CALL(funcname)
dnl
dnl  Simply override the definition in x86_64-defs.m4.

define(`CALL',`call	GSYM_PREFIX`'$1')


dnl  Usage: JUMPTABSECT
dnl
dnl  CAUTION: Do not put anything sensible here, like RODATA.  That works with
dnl  some Darwin tool chains, but silently breaks with other.  (Note that
dnl  putting jump tables in the text segment is a really poor idea for PC many
dnl  processors, since they cannot cache the same thing in both L1D and L2I.)

define(`JUMPTABSECT', `.text')


dnl  Usage: JMPENT(targlabel,tablabel)

define(`JMPENT',`dnl
ifdef(`PIC',
	`.set	$1_tmp, $1-$2
	.long	$1_tmp'
,
	`.quad	$1'
)')

dnl  Target ABI macros.  For Darwin we override IFELF (and leave default for
dnl  IFDOS and IFSTD).

define(`IFELF',   `')


dnl  Usage: PROTECT(symbol)
dnl
dnl  Used for private GMP symbols that should never be overridden by users.
dnl  This can save reloc entries and improve shlib sharing as well as
dnl  application startup times

define(`PROTECT',  `.private_extern $1')


divert`'dnl
