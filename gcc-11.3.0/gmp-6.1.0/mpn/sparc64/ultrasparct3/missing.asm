dnl  SPARC v9-2011 simulation support.

dnl  Contributed to the GNU project by Torbjörn Granlund.

dnl  Copyright 2013 Free Software Foundation, Inc.

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

ASM_START()
PROLOGUE(__gmpn_umulh)
	save	%sp, -176, %sp
	ldx	[%sp+2047+176+256], %o0
	ldx	[%sp+2047+176+256+8], %o1
	rd	%ccr, %o4
	srl	%o0, 0, %l4
	srl	%o1, 0, %l1
	srlx	%o1, 32, %o1
	mulx	%o1, %l4, %l2
	srlx	%o0, 32, %o0
	mulx	%o0, %l1, %l3
	mulx	%l1, %l4, %l1
	srlx	%l1, 32, %l1
	add	%l2, %l1, %l2
	addcc	%l2, %l3, %l2
	mulx	%o1, %o0, %o1
	mov	0, %l1
	movcs	%xcc, 1, %l1
	sllx	%l1, 32, %l1
	add	%o1, %l1, %o1
	srlx	%l2, 32, %o0
	add	%o1, %o0, %o0
	stx	%o0, [%sp+2047+176+256]
	wr	%o4, 0, %ccr
	ret
	 restore
EPILOGUE()

PROLOGUE(__gmpn_lzcnt)
	save	%sp, -176, %sp
	ldx	[%sp+2047+176+256], %o0
	brz,a	%o0, 2f
	 mov	64, %o1
	brlz	%o0, 2f
	 mov	0, %o1
1:	sllx	%o0, 1, %o0
	brgz	%o0, 1b
	 add	%o1, 1, %o1
	stx	%o1, [%sp+2047+176+256]
2:	ret
	 restore
EPILOGUE()
