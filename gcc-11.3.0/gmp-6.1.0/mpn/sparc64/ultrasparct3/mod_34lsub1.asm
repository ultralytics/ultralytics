dnl  SPARC v9 mpn_mod_34lsub1 for T3/T4/T5.

dnl  Copyright 2005, 2013 Free Software Foundation, Inc.

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

C		    cycles/limb
C UltraSPARC T1:	 -
C UltraSPARC T3:	 5
C UltraSPARC T4:	 1.57

C This is based on the powerpc64/mode64 code.

C INPUT PARAMETERS
define(`up', `%i0')
define(`n',  `%i1')

ASM_START()
	REGISTER(%g2,#scratch)
	REGISTER(%g3,#scratch)
PROLOGUE(mpn_mod_34lsub1)
	save	%sp, -176, %sp

	mov	0, %g1
	mov	0, %g3
	mov	0, %g4
	addcc	%g0, 0, %g5

	add	n, -3, n
	brlz	n, L(lt3)
	 nop

	add	n, -3, n
	ldx	[up+0], %l5
	ldx	[up+8], %l6
	ldx	[up+16], %l7
	brlz	n, L(end)
	 add	up, 24, up

	ALIGN(16)
L(top):	addxccc(%g1, %l5, %g1)
	ldx	[up+0], %l5
	addxccc(%g3, %l6, %g3)
	ldx	[up+8], %l6
	addxccc(%g4, %l7, %g4)
	ldx	[up+16], %l7
	add	n, -3, n
	brgez	n, L(top)
	 add	up, 24, up

L(end):	addxccc(	%g1, %l5, %g1)
	addxccc(%g3, %l6, %g3)
	addxccc(%g4, %l7, %g4)
	addxc(	%g5, %g0, %g5)

L(lt3):	cmp	n, -2
	blt	L(2)
	 nop

	ldx	[up+0], %l5
	mov	0, %l6
	beq	L(1)
	 addcc	%g1, %l5, %g1

	ldx	[up+8], %l6
L(1):	addxccc(%g3, %l6, %g3)
	addxccc(%g4, %g0, %g4)
	addxc(	%g5, %g0, %g5)

L(2):	sllx	%g1, 16, %l0
	srlx	%l0, 16, %l0		C %l0 = %g1 mod 2^48
	srlx	%g1, 48, %l3		C %l3 = %g1 div 2^48
	srl	%g3, 0, %g1
	sllx	%g1, 16, %l4		C %l4 = (%g3 mod 2^32) << 16
	srlx	%g3, 32, %l5		C %l5 = %g3 div 2^32
	sethi	%hi(0xffff0000), %g1
	andn	%g4, %g1, %g1
	sllx	%g1, 32, %l6		C %l6 = (%g4 mod 2^16) << 32
	srlx	%g4, 16, %l7		C %l7 = %g4 div 2^16

	add	%l0, %l3, %l0
	add	%l4, %l5, %l4
	add	%l6, %l7, %l6

	add	%l0, %l4, %l0
	add	%l6, %g5, %l6

	add	%l0, %l6, %i0
	ret
	 restore
EPILOGUE()
