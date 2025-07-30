dnl  S/390-64 mpn_add_n and mpn_sub_n.

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
C z900		 5.5
C z990		 3
C z9		 ?
C z10		 6
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
  define(ADSB,		alg)
  define(ADSBCR,	alcgr)
  define(ADSBC,		alcg)
  define(RETVAL,`dnl
	lghi	%r2, 0
	alcgr	%r2, %r2')
  define(func,		mpn_add_n)
  define(func_nc,	mpn_add_nc)')
ifdef(`OPERATION_sub_n', `
  define(ADSB,		slg)
  define(ADSBCR,	slbgr)
  define(ADSBC,		slbg)
  define(RETVAL,`dnl
	slbgr	%r2, %r2
	lcgr	%r2, %r2')
  define(func,		mpn_sub_n)
  define(func_nc,	mpn_sub_nc)')

MULFUNC_PROLOGUE(mpn_add_n mpn_sub_n)

ASM_START()
PROLOGUE(func)
	stmg	%r6, %r8, 48(%r15)

	aghi	n, 3
	lghi	%r7, 3
	srlg	%r1, n, 2
	ngr	%r7, n			C n mod 4
	je	L(b1)
	cghi	%r7, 2
	jl	L(b2)
	jne	L(b0)

L(b3):	lmg	%r5, %r7, 0(up)
	la	up, 24(up)
	ADSB	%r5, 0(vp)
	ADSBC	%r6, 8(vp)
	ADSBC	%r7, 16(vp)
	la	vp, 24(vp)
	stmg	%r5, %r7, 0(rp)
	la	rp, 24(rp)
	brctg	%r1, L(top)
	j	L(end)

L(b0):	lmg	%r5, %r8, 0(up)		C This redundant insns is no mistake,
	la	up, 32(up)		C it is needed to make main loop run
	ADSB	%r5, 0(vp)		C fast for n = 0 (mod 4).
	ADSBC	%r6, 8(vp)
	j	L(m0)

L(b1):	lg	%r5, 0(up)
	la	up, 8(up)
	ADSB	%r5, 0(vp)
	la	vp, 8(vp)
	stg	%r5, 0(rp)
	la	rp, 8(rp)
	brctg	%r1, L(top)
	j	L(end)

L(b2):	lmg	%r5, %r6, 0(up)
	la	up, 16(up)
	ADSB	%r5, 0(vp)
	ADSBC	%r6, 8(vp)
	la	vp, 16(vp)
	stmg	%r5, %r6, 0(rp)
	la	rp, 16(rp)
	brctg	%r1, L(top)
	j	L(end)

L(top):	lmg	%r5, %r8, 0(up)
	la	up, 32(up)
	ADSBC	%r5, 0(vp)
	ADSBC	%r6, 8(vp)
L(m0):	ADSBC	%r7, 16(vp)
	ADSBC	%r8, 24(vp)
	la	vp, 32(vp)
	stmg	%r5, %r8, 0(rp)
	la	rp, 32(rp)
	brctg	%r1, L(top)

L(end):	RETVAL
	lmg	%r6, %r8, 48(%r15)
	br	%r14
EPILOGUE()
