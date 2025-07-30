dnl  S/390-64 mpn_addlsh1_n and mpn_rsblsh1_n.

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
C z900		 9
C z990		 4.75
C z9		 ?
C z10		11
C z196		 ?

C TODO
C  * Optimise for small n, avoid 'la' like in aors_n.asm.
C  * Tune to reach 3.5 c/l.  For addlsh1, we could let the main alcgr propagate
C    carry to the lsh1 alcgr.
C  * Compute RETVAL for sublsh1_n less stupidly.

C INPUT PARAMETERS
define(`rp',	`%r2')
define(`up',	`%r3')
define(`vp',	`%r4')
define(`n',	`%r5')

ifdef(`OPERATION_addlsh1_n',`
  define(ADSB,		alg)
  define(ADSBC,		alcg)
  define(INITCY,	`lghi	%r9, -1')
  define(RETVAL,	`la	%r2, 2(%r1,%r9)')
  define(func, mpn_addlsh1_n)
')
ifdef(`OPERATION_rsblsh1_n',`
  define(ADSB,		slg)
  define(ADSBC,		slbg)
  define(INITCY,	`lghi	%r9, 0')
  define(RETVAL,`dnl
	algr	%r1, %r9
	lghi	%r2, 1
	algr	%r2, %r1')
  define(func, mpn_rsblsh1_n)
')

MULFUNC_PROLOGUE(mpn_addlsh1_n mpn_rsblsh1_n)

ASM_START()
PROLOGUE(func)
	stmg	%r6, %r9, 48(%r15)

	aghi	n, 3
	lghi	%r7, 3
	srlg	%r0, n, 2
	ngr	%r7, n			C n mod 4
	je	L(b1)
	cghi	%r7, 2
	jl	L(b2)
	jne	L(b0)

L(b3):	lmg	%r5, %r7, 0(vp)
	la	vp, 24(vp)

	algr	%r5, %r5
	alcgr	%r6, %r6
	alcgr	%r7, %r7
	slbgr	%r1, %r1

	ADSB	%r5, 0(up)
	ADSBC	%r6, 8(up)
	ADSBC	%r7, 16(up)
	la	up, 24(up)
	slbgr	%r9, %r9

	stmg	%r5, %r7, 0(rp)
	la	rp, 24(rp)
	brctg	%r0, L(top)
	j	L(end)

L(b0):	lghi	%r1, -1
	INITCY
	j	L(top)

L(b1):	lg	%r5, 0(vp)
	la	vp, 8(vp)

	algr	%r5, %r5
	slbgr	%r1, %r1
	ADSB	%r5, 0(up)
	la	up, 8(up)
	slbgr	%r9, %r9

	stg	%r5, 0(rp)
	la	rp, 8(rp)
	brctg	%r0, L(top)
	j	L(end)

L(b2):	lmg	%r5, %r6, 0(vp)
	la	vp, 16(vp)

	algr	%r5, %r5
	alcgr	%r6, %r6
	slbgr	%r1, %r1

	ADSB	%r5, 0(up)
	ADSBC	%r6, 8(up)
	la	up, 16(up)
	slbgr	%r9, %r9

	stmg	%r5, %r6, 0(rp)
	la	rp, 16(rp)
	brctg	%r0, L(top)
	j	L(end)

L(top):	lmg	%r5, %r8, 0(vp)
	la	vp, 32(vp)

	aghi	%r1, 1			C restore carry

	alcgr	%r5, %r5
	alcgr	%r6, %r6
	alcgr	%r7, %r7
	alcgr	%r8, %r8

	slbgr	%r1, %r1		C save carry

	aghi	%r9, 1			C restore carry

	ADSBC	%r5, 0(up)
	ADSBC	%r6, 8(up)
	ADSBC	%r7, 16(up)
	ADSBC	%r8, 24(up)
	la	up, 32(up)

	slbgr	%r9, %r9		C save carry

	stmg	%r5, %r8, 0(rp)
	la	rp, 32(rp)
	brctg	%r0, L(top)

L(end):	RETVAL
	lmg	%r6, %r9, 48(%r15)
	br	%r14
EPILOGUE()
