dnl  SPARC v9 mpn_lshiftc

dnl  Copyright 1996, 2000-2003, 2010 Free Software Foundation, Inc.

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
C UltraSPARC 1&2:     3
C UltraSPARC 3:	      2.67

C INPUT PARAMETERS
define(`rp', `%i0')
define(`up', `%i1')
define(`n',  `%i2')
define(`cnt',`%i3')

define(`u0', `%l0')
define(`u1', `%l2')
define(`u2', `%l4')
define(`u3', `%l6')

define(`tnc',`%i4')

define(`fanop',`fitod %f0,%f2')		dnl  A quasi nop running in the FA pipe

ASM_START()
	REGISTER(%g2,#scratch)
	REGISTER(%g3,#scratch)
PROLOGUE(mpn_lshiftc)
	save	%sp,-160,%sp

	sllx	n,3,%g1
	sub	%g0,cnt,tnc		C negate shift count
	add	up,%g1,up		C make %o1 point at end of src
	add	rp,%g1,rp		C make %o0 point at end of res
	ldx	[up-8],u3		C load first limb
	subcc	n,5,n
	srlx	u3,tnc,%i5		C compute function result
	bl,pn	%xcc,.Lend1234
	sllx	u3,cnt,%g3

	subcc	n,4,n
	ldx	[up-16],u0
	ldx	[up-24],u1
	add	up,-32,up
	ldx	[up-0],u2
	ldx	[up-8],u3
	srlx	u0,tnc,%g2
	bl,pn	%xcc,.Lend5678
	not	%g3, %g3

	b,a	.Loop
	ALIGN(16)
.Loop:
	sllx	u0,cnt,%g1
	andn	%g3,%g2,%g3
	ldx	[up-16],u0
	fanop
C --
	srlx	u1,tnc,%g2
	subcc	n,4,n
	stx	%g3,[rp-8]
	not	%g1, %g1
C --
	sllx	u1,cnt,%g3
	andn	%g1,%g2,%g1
	ldx	[up-24],u1
	fanop
C --
	srlx	u2,tnc,%g2
	stx	%g1,[rp-16]
	add	up,-32,up
	not	%g3, %g3
C --
	sllx	u2,cnt,%g1
	andn	%g3,%g2,%g3
	ldx	[up-0],u2
	fanop
C --
	srlx	u3,tnc,%g2
	stx	%g3,[rp-24]
	add	rp,-32,rp
	not	%g1, %g1
C --
	sllx	u3,cnt,%g3
	andn	%g1,%g2,%g1
	ldx	[up-8],u3
	fanop
C --
	srlx	u0,tnc,%g2
	stx	%g1,[rp-0]
	bge,pt	%xcc,.Loop
	not	%g3, %g3
C --
.Lend5678:
	sllx	u0,cnt,%g1
	andn	%g3,%g2,%g3
	srlx	u1,tnc,%g2
	stx	%g3,[rp-8]
	not	%g1, %g1
	sllx	u1,cnt,%g3
	andn	%g1,%g2,%g1
	srlx	u2,tnc,%g2
	stx	%g1,[rp-16]
	not	%g3, %g3
	sllx	u2,cnt,%g1
	andn	%g3,%g2,%g3
	srlx	u3,tnc,%g2
	stx	%g3,[rp-24]
	add	rp,-32,rp
	not	%g1, %g1
	sllx	u3,cnt,%g3		C carry...
	andn	%g1,%g2,%g1
	stx	%g1,[rp-0]

.Lend1234:
	addcc	n,4,n
	bz,pn	%xcc,.Lret
	fanop
.Loop0:
	add	rp,-8,rp
	subcc	n,1,n
	ldx	[up-16],u3
	add	up,-8,up
	srlx	u3,tnc,%g2
	not	%g3, %g3
	andn	%g3,%g2,%g3
	stx	%g3,[rp]
	sllx	u3,cnt,%g3
	bnz,pt	%xcc,.Loop0
	fanop
.Lret:
	not	%g3, %g3
	stx	%g3,[rp-8]
	mov	%i5,%i0
	ret
	restore
EPILOGUE()
