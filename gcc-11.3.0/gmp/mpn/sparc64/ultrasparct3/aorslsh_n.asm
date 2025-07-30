dnl  SPARC v9 mpn_addlsh_n and mpn_sublsh_n for T3/T4/T5.

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

C		   cycles/limb
C UltraSPARC T3:	11
C UltraSPARC T4:	 4

C For sublsh_n we combine the two shifted limbs using xnor, using the identity
C (a xor not b) = (not (a xor b)) which equals (not (a or b)) when (a and b) =
C 0 as it is in our usage.  This gives us the ones complement for free.
C Unfortunately, the same trick will not work for rsblsh_n, which will instead
C require a separate negation.
C
C FIXME: Add rsblsh_n to this file.

define(`rp', `%i0')
define(`up', `%i1')
define(`vp', `%i2')
define(`n',  `%i3')
define(`cnt',`%i4')

define(`tnc',`%o5')

ifdef(`OPERATION_addlsh_n',`
  define(`INITCY', `subcc	%g0, 0, %g0')
  define(`MERGE',  `or')
  define(`func',   `mpn_addlsh_n')
')
ifdef(`OPERATION_sublsh_n',`
  define(`INITCY', `subcc	%g0, 1, %g0')
  define(`MERGE',  `xnor')
  define(`func',   `mpn_sublsh_n')
')

define(`rp0',  `rp')
define(`rp1',  `%o2')
define(`up0',  `up')
define(`up1',  `%o3')
define(`vp0',  `vp')
define(`vp1',  `%o4')

MULFUNC_PROLOGUE(mpn_addlsh_n mpn_sublsh_n)
ASM_START()
	REGISTER(%g2,#scratch)
	REGISTER(%g3,#scratch)
PROLOGUE(func)
	save	%sp, -176, %sp
	mov	64, tnc
	sub	tnc, cnt, tnc

	andcc	n, 1, %g0
	sllx	n, 3, n
	add	n, -16, n
	add	up, n, up0
	add	vp, n, vp0
	add	rp, n, rp0
	add	up0, 8, up1
	add	vp0, 8, vp1
	add	rp0, -8, rp1
	add	rp0, -16, rp0
	neg	n, n
	be	L(evn)
	 INITCY

L(odd):	ldx	[vp0 + n], %l1
	mov	0, %l2
	ldx	[up0 + n], %l5
	sllx	%l1, cnt, %g3
	brgez	n, L(wd1)
	 add	n, 8, n
	ldx	[vp0 + n], %l0
	b	L(lo1)
	 sllx	%l1, cnt, %g3

L(evn):	ldx	[vp0 + n], %l0
	mov	0, %l3
	ldx	[up0 + n], %l4
	ldx	[vp1 + n], %l1
	b	L(lo0)
	 sllx	%l0, cnt, %g1

L(top):	addxccc(%l6, %l4, %o0)
	ldx	[vp0 + n], %l0
	sllx	%l1, cnt, %g3
	stx	%o0, [rp0 + n]
L(lo1):	srlx	%l1, tnc, %l3
	MERGE	%l2, %g3, %l7
	ldx	[up0 + n], %l4
	addxccc(%l7, %l5, %o1)
	ldx	[vp1 + n], %l1
	sllx	%l0, cnt, %g1
	stx	%o1, [rp1 + n]
L(lo0):	srlx	%l0, tnc, %l2
	MERGE	%l3, %g1, %l6
	ldx	[up1 + n], %l5
	brlz,pt	n, L(top)
	 add	n, 16, n

	addxccc(%l6, %l4, %o0)
	sllx	%l1, cnt, %g3
	stx	%o0, [rp0 + n]
L(wd1):	srlx	%l1, tnc, %l3
	MERGE	%l2, %g3, %l7
	addxccc(%l7, %l5, %o1)
	stx	%o1, [rp1 + n]

ifdef(`OPERATION_addlsh_n',
`	addxc(	%l3, %g0, %i0)')
ifdef(`OPERATION_sublsh_n',
`	addxc(	%g0, %g0, %g1)
	add	%g1, -1, %g1
	sub	%l3, %g1, %i0')

	ret
	 restore
EPILOGUE()
