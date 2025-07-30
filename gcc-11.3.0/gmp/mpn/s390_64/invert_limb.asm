dnl  S/390-64 mpn_invert_limb

dnl  Contributed to the GNU project by Torbjorn Granlund.

dnl  Copyright 2011, 2013 Free Software Foundation, Inc.

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
C z900	       142
C z990          86
C z9		 ?
C z10	       120
C z196		 ?

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_invert_limb)
	stg	%r9, 72(%r15)
	srlg	%r9, %r2, 55
	agr	%r9, %r9
	larl	%r4, approx_tab-512
	srlg	%r3, %r2, 24
	aghi	%r3, 1
	lghi	%r5, 1
	llgh	%r4, 0(%r9, %r4)
	sllg	%r9, %r4, 11
	msgr	%r4, %r4
	msgr	%r4, %r3
	srlg	%r4, %r4, 40
	aghi	%r9, -1
	sgr	%r9, %r4
	sllg	%r0, %r9, 60
	sllg	%r1, %r9, 13
	msgr	%r9, %r9
	msgr	%r9, %r3
	sgr	%r0, %r9
	ngr	%r5, %r2
	srlg	%r4, %r2, 1
	srlg	%r3, %r0, 47
	agr	%r3, %r1
	agr	%r4, %r5
	msgr	%r4, %r3
	srlg	%r1, %r3, 1
	lcgr	%r5, %r5
	ngr	%r1, %r5
	sgr	%r1, %r4
	mlgr	%r0, %r3
	srlg	%r9, %r0, 1
	sllg	%r4, %r3, 31
	agr	%r4, %r9
	lgr	%r1, %r4
	mlgr	%r0, %r2
	algr	%r1, %r2
	alcgr	%r0, %r2
	lgr	%r2, %r4
	sgr	%r2, %r0
	lg	%r9, 72(%r15)
	br	%r14
EPILOGUE()
	RODATA
	ALIGN(2)
approx_tab:
forloop(i,256,512-1,dnl
`	.word	eval(0x7fd00/i)
')dnl
ASM_END()
