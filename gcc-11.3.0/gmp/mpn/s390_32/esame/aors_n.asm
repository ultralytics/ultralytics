dnl  S/390-32 mpn_add_n and mpn_sub_n.

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
C z900		 ?
C z990	      2.75-3		(fast for even n, slow for odd n)
C z9		 ?
C z10		 ?
C z196		 ?

C TODO
C  * Optimise for small n
C  * Use r0 and save/restore one less register
C  * Using logops_n's v1 inner loop operand order make the loop about 20%
C    faster, at the expense of highly alignment-dependent performance.

C INPUT PARAMETERS
define(`rp',	`%r2')
define(`up',	`%r3')
define(`vp',	`%r4')
define(`n',	`%r5')

ifdef(`OPERATION_add_n', `
  define(ADSB,		al)
  define(ADSBCR,	alcr)
  define(ADSBC,		alc)
  define(RETVAL,`dnl
	lhi	%r2, 0
	alcr	%r2, %r2')
  define(func,		mpn_add_n)
  define(func_nc,	mpn_add_nc)')
ifdef(`OPERATION_sub_n', `
  define(ADSB,		sl)
  define(ADSBCR,	slbr)
  define(ADSBC,		slb)
  define(RETVAL,`dnl
	slbr	%r2, %r2
	lcr	%r2, %r2')
  define(func,		mpn_sub_n)
  define(func_nc,	mpn_sub_nc)')

MULFUNC_PROLOGUE(mpn_add_n mpn_sub_n)

ASM_START()
PROLOGUE(func)
	stm	%r6, %r8, 24(%r15)

	ahi	n, 3
	lhi	%r7, 3
	lr	%r1, n
	srl	%r1, 2
	nr	%r7, n			C n mod 4
	je	L(b1)
	chi	%r7, 2
	jl	L(b2)
	jne	L(b0)

L(b3):	lm	%r5, %r7, 0(up)
	la	up, 12(up)
	ADSB	%r5, 0(vp)
	ADSBC	%r6, 4(vp)
	ADSBC	%r7, 8(vp)
	la	vp, 12(vp)
	stm	%r5, %r7, 0(rp)
	la	rp, 12(rp)
	brct	%r1, L(top)
	j	L(end)

L(b0):	lm	%r5, %r8, 0(up)		C This redundant insns is no mistake,
	la	up, 16(up)		C it is needed to make main loop run
	ADSB	%r5, 0(vp)		C fast for n = 0 (mod 4).
	ADSBC	%r6, 4(vp)
	j	L(m0)

L(b1):	l	%r5, 0(up)
	la	up, 4(up)
	ADSB	%r5, 0(vp)
	la	vp, 4(vp)
	st	%r5, 0(rp)
	la	rp, 4(rp)
	brct	%r1, L(top)
	j	L(end)

L(b2):	lm	%r5, %r6, 0(up)
	la	up, 8(up)
	ADSB	%r5, 0(vp)
	ADSBC	%r6, 4(vp)
	la	vp, 8(vp)
	stm	%r5, %r6, 0(rp)
	la	rp, 8(rp)
	brct	%r1, L(top)
	j	L(end)

L(top):	lm	%r5, %r8, 0(up)
	la	up, 16(up)
	ADSBC	%r5, 0(vp)
	ADSBC	%r6, 4(vp)
L(m0):	ADSBC	%r7, 8(vp)
	ADSBC	%r8, 12(vp)
	la	vp, 16(vp)
	stm	%r5, %r8, 0(rp)
	la	rp, 16(rp)
	brct	%r1, L(top)

L(end):	RETVAL
	lm	%r6, %r8, 24(%r15)
	br	%r14
EPILOGUE()
