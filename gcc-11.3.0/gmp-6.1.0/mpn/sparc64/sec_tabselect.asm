dnl  SPARC v9 mpn_sec_tabselect.

dnl  Contributed to the GNU project by Torbjörn Granlund and David Miller.

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
C UltraSPARC 1&2:	 2 hopefully
C UltraSPARC 3:		 3
C UltraSPARC T1:	17
C UltraSPARC T3:	 ?
C UltraSPARC T4/T5:	 2.25 hopefully

C INPUT PARAMETERS
define(`rp',     `%i0')
define(`tp',     `%i1')
define(`n',      `%i2')
define(`nents',  `%i3')
define(`which',  `%i4')

define(`i',      `%g1')
define(`j',      `%g3')
define(`stride', `%g4')
define(`tporig', `%g5')
define(`mask',   `%o0')

define(`data0',  `%l0')
define(`data1',  `%l1')
define(`data2',  `%l2')
define(`data3',  `%l3')
define(`t0',     `%l4')
define(`t1',     `%l5')
define(`t2',     `%l6')
define(`t3',     `%l7')

ASM_START()
	REGISTER(%g2,#scratch)
	REGISTER(%g3,#scratch)
PROLOGUE(mpn_sec_tabselect)
	save	%sp, -176, %sp

	sllx	n, 3, stride
	sub	n, 4, j
	brlz	j, L(outer_end)
	 mov	tp, tporig

L(outer_loop):
	clr	data0
	clr	data1
	clr	data2
	clr	data3
	mov	tporig, tp
	mov	nents, i
	mov	which, %o1

L(top):	subcc	%o1, 1, %o1		C set carry iff o1 = 0
	ldx	[tp + 0], t0
	subc	%g0, %g0, mask
	ldx	[tp + 8], t1
	sub	i, 1, i
	ldx	[tp + 16], t2
	ldx	[tp + 24], t3
	add	tp, stride, tp
	and	t0, mask, t0
	and	t1, mask, t1
	or	t0, data0, data0
	and	t2, mask, t2
	or	t1, data1, data1
	and	t3, mask, t3
	or	t2, data2, data2
	brnz	i, L(top)
	 or	t3, data3, data3

	stx	data0, [rp + 0]
	subcc	j, 4, j
	stx	data1, [rp + 8]
	stx	data2, [rp + 16]
	stx	data3, [rp + 24]
	add	tporig, (4 * 8), tporig

	brgez	j, L(outer_loop)
	 add	rp, (4 * 8), rp
L(outer_end):


	andcc	n, 2, %g0
	be	L(b0x)
	 nop
L(b1x):	clr	data0
	clr	data1
	mov	tporig, tp
	mov	nents, i
	mov	which, %o1

L(tp2):	subcc	%o1, 1, %o1
	ldx	[tp + 0], t0
	subc	%g0, %g0, mask
	ldx	[tp + 8], t1
	sub	i, 1, i
	add	tp, stride, tp
	and	t0, mask, t0
	and	t1, mask, t1
	or	t0, data0, data0
	brnz	i, L(tp2)
	 or	t1, data1, data1

	stx	data0, [rp + 0]
	stx	data1, [rp + 8]
	add	tporig, (2 * 8), tporig
	add	rp, (2 * 8), rp


L(b0x):	andcc	n, 1, %g0
	be	L(b00)
	 nop
L(b01):	clr	data0
	mov	tporig, tp
	mov	nents, i
	mov	which, %o1

L(tp1):	subcc	%o1, 1, %o1
	ldx	[tp + 0], t0
	subc	%g0, %g0, mask
	sub	i, 1, i
	add	tp, stride, tp
	and	t0, mask, t0
	brnz	i, L(tp1)
	 or	t0, data0, data0

	stx	data0, [rp + 0]

L(b00):	 ret
	  restore
EPILOGUE()
