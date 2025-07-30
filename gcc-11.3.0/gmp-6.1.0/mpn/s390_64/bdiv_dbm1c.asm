dnl  S/390-64 mpn_bdiv_dbm1c

dnl  Copyright 2011 Free Software Foundation, Inc.

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

C            cycles/limb
C z900		29
C z990		22
C z9		 ?
C z10		19
C z196		 ?

C INPUT PARAMETERS
define(`qp',	  `%r2')
define(`up',	  `%r3')
define(`n',	  `%r4')
define(`bd',	  `%r5')
define(`cy',	  `%r6')

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_bdiv_dbm1c)
	stmg	%r6, %r7, 48(%r15)
	lghi	%r7, 0			C zero index register

L(top):	lg	%r1, 0(%r7,up)
	mlgr	%r0, bd
	slgr	%r6, %r1
	stg	%r6, 0(%r7,qp)
	la	%r7, 8(%r7)
	slbgr	%r6, %r0
	brctg	n, L(top)

	lgr	%r2, %r6
	lmg	%r6, %r7, 48(%r15)
	br	%r14
EPILOGUE()
