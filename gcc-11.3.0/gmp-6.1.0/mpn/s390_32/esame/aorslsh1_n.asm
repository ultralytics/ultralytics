dnl  S/390-32 mpn_addlsh1_n

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
C z900		 9.25
C z990		 5
C z9		 ?
C z10		 ?
C z196		 ?

C TODO
C  * Optimise for small n
C  * Compute RETVAL for sublsh1_n less stupidly

C INPUT PARAMETERS
define(`rp',	`%r2')
define(`up',	`%r3')
define(`vp',	`%r4')
define(`n',	`%r5')

ifdef(`OPERATION_addlsh1_n',`
  define(ADDSUBC,       alr)
  define(ADDSUBE,       alcr)
  define(INITCY,        `lhi	%r13, -1')
  define(RETVAL,        `alr	%r1, %r13
			lhi	%r2, 2
			alr	%r2, %r1')
  define(func, mpn_addlsh1_n)
')
ifdef(`OPERATION_sublsh1_n',`
  define(ADDSUBC,       slr)
  define(ADDSUBE,       slbr)
  define(INITCY,        `lhi	%r13, 0')
  define(RETVAL,        `slr	%r1, %r13
			lhi	%r2, 1
			alr	%r2, %r1')
  define(func, mpn_sublsh1_n)
')

MULFUNC_PROLOGUE(mpn_addlsh1_n mpn_sublsh1_n)

ASM_START()
PROLOGUE(func)
	stm	%r6, %r13, 24(%r15)

	la	%r0, 3(n)
	lhi	%r7, 3
	srl	%r0, 2
	nr	%r7, n			C n mod 4
	je	L(b0)
	chi	%r7, 2
	jl	L(b1)
	je	L(b2)

L(b3):	lm	%r5, %r7, 0(up)
	la	up, 12(up)
	lm	%r9, %r11, 0(vp)
	la	vp, 12(vp)

	alr	%r9, %r9
	alcr	%r10, %r10
	alcr	%r11, %r11
	slbr	%r1, %r1

	ADDSUBC	%r5, %r9
	ADDSUBE	%r6, %r10
	ADDSUBE	%r7, %r11
	slbr	%r13, %r13

	stm	%r5, %r7, 0(rp)
	la	rp, 12(rp)
	brct	%r0, L(top)
	j	L(end)

L(b0):	lhi	%r1, -1
	INITCY
	j	L(top)

L(b1):	l	%r5, 0(up)
	la	up, 4(up)
	l	%r9, 0(vp)
	la	vp, 4(vp)

	alr	%r9, %r9
	slbr	%r1, %r1
	ADDSUBC	%r5, %r9
	slbr	%r13, %r13

	st	%r5, 0(rp)
	la	rp, 4(rp)
	brct	%r0, L(top)
	j	L(end)

L(b2):	lm	%r5, %r6, 0(up)
	la	up, 8(up)
	lm	%r9, %r10, 0(vp)
	la	vp, 8(vp)

	alr	%r9, %r9
	alcr	%r10, %r10
	slbr	%r1, %r1

	ADDSUBC	%r5, %r9
	ADDSUBE	%r6, %r10
	slbr	%r13, %r13

	stm	%r5, %r6, 0(rp)
	la	rp, 8(rp)
	brct	%r0, L(top)
	j	L(end)

L(top):	lm	%r9, %r12, 0(vp)
	la	vp, 16(vp)

	ahi	%r1, 1			C restore carry

	alcr	%r9, %r9
	alcr	%r10, %r10
	alcr	%r11, %r11
	alcr	%r12, %r12

	slbr	%r1, %r1		C save carry

	lm	%r5, %r8, 0(up)
	la	up, 16(up)

	ahi	%r13, 1			C restore carry

	ADDSUBE	%r5, %r9
	ADDSUBE	%r6, %r10
	ADDSUBE	%r7, %r11
	ADDSUBE	%r8, %r12

	slbr	%r13, %r13

	stm	%r5, %r8, 0(rp)
	la	rp, 16(rp)
	brct	%r0, L(top)

L(end):
	RETVAL
	lm	%r6, %r13, 24(%r15)
	br	%r14
EPILOGUE()
