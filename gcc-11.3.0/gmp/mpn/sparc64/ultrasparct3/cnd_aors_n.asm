dnl  SPARC v9 mpn_cnd_add_n and mpn_cnd_sub_n for T3/T4/T5.

dnl  Contributed to the GNU project by David Miller and Torbjörn Granlund.

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

C		   cycles/limb
C UltraSPARC T3:	 8.5
C UltraSPARC T4:	 3

C We use a double-pointer trick to allow indexed addressing.  Its setup
C cost might be a problem in these functions, since we don't expect huge n
C arguments.
C
C For sub we need ~(a & mask) = (~a | ~mask) but by complementing mask we can
C instead do ~(a & ~mask) = (~a | mask), allowing us to use the orn insn.

C INPUT PARAMETERS
define(`cnd', `%i0')
define(`rp',  `%i1')
define(`up',  `%i2')
define(`vp',  `%i3')
define(`n',   `%i4')

define(`mask',   `cnd')
define(`up0', `%l0')  define(`up1', `%l1')
define(`vp0', `%l2')  define(`vp1', `%l3')
define(`rp0', `%g4')  define(`rp1', `%g5')
define(`u0',  `%l4')  define(`u1',  `%l5')
define(`v0',  `%l6')  define(`v1',  `%l7')
define(`x0',  `%g1')  define(`x1',  `%g3')
define(`w0',  `%g1')  define(`w1',  `%g3')

ifdef(`OPERATION_cnd_add_n',`
  define(`LOGOP',   `and	$1, $2, $3')
  define(`MAKEMASK',`cmp	%g0, $1
		     subc	%g0, %g0, $2')
  define(`INITCY',  `addcc	%g0, 0, %g0')
  define(`RETVAL',  `addxc(	%g0, %g0, %i0)')
  define(`func',    `mpn_cnd_add_n')
')
ifdef(`OPERATION_cnd_sub_n',`
  define(`LOGOP',   `orn	$2, $1, $3')
  define(`MAKEMASK',`cmp	$1, 1
		     subc	%g0, %g0, $2')
  define(`INITCY',  `subcc	%g0, 1, %g0')
  define(`RETVAL',  `addxc(	%g0, %g0, %i0)
		     xor	%i0, 1, %i0')
  define(`func',    `mpn_cnd_sub_n')
')

MULFUNC_PROLOGUE(mpn_cnd_add_n mpn_cnd_sub_n)

ASM_START()
	REGISTER(%g2,#scratch)
	REGISTER(%g3,#scratch)
PROLOGUE(func)
	save	%sp, -176, %sp

	MAKEMASK(cnd,mask)

	andcc	n, 1, %g0
	sllx	n, 3, n
	add	n, -16, n
	add	vp, n, vp0
	add	up, n, up0
	add	rp, n, rp0
	neg	n, n
	be	L(evn)
	 INITCY

L(odd):	ldx	[vp0 + n], v1
	ldx	[up0 + n], u1
	LOGOP(	v1, mask, x1)
	addxccc(u1, x1, w1)
	stx	w1, [rp0 + n]
	add	n, 8, n
	brgz	n, L(rtn)
	 nop

L(evn):	add	vp0, 8, vp1
	add	up0, 8, up1
	add	rp0, -24, rp1
	ldx	[vp0 + n], v0
	ldx	[vp1 + n], v1
	ldx	[up0 + n], u0
	ldx	[up1 + n], u1
	add	n, 16, n
	brgz	n, L(end)
	 add	rp0, -16, rp0

L(top):	LOGOP(	v0, mask, x0)
	ldx	[vp0 + n], v0
	LOGOP(	v1, mask, x1)
	ldx	[vp1 + n], v1
	addxccc(u0, x0, w0)
	ldx	[up0 + n], u0
	addxccc(u1, x1, w1)
	ldx	[up1 + n], u1
	stx	w0, [rp0 + n]
	add	n, 16, n
	brlez	n, L(top)
	 stx	w1, [rp1 + n]

L(end):	LOGOP(	v0, mask, x0)
	LOGOP(	v1, mask, x1)
	addxccc(u0, x0, w0)
	addxccc(u1, x1, w1)
	stx	w0, [rp0 + n]
	stx	w1, [rp1 + 32]

L(rtn):	RETVAL
	ret
	 restore
EPILOGUE()
