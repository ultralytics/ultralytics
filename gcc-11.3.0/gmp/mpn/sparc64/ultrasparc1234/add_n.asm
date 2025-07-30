dnl  SPARC v9 mpn_add_n -- Add two limb vectors of the same length > 0 and
dnl  store sum in a third limb vector.

dnl  Copyright 2001-2003, 2011 Free Software Foundation, Inc.

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
C UltraSPARC 1&2:     4
C UltraSPARC 3:	      4.5

C Compute carry-out from the most significant bits of u,v, and r, where
C r=u+v+carry_in, using logic operations.

C This code runs at 4 cycles/limb on UltraSPARC 1 and 2.  It has a 4 insn
C recurrency, and the UltraSPARC 1 and 2 the IE units are 100% saturated.
C Therefore, it seems futile to try to optimize this any further...

C INPUT PARAMETERS
define(`rp', `%i0')
define(`up', `%i1')
define(`vp', `%i2')
define(`n',  `%i3')

define(`u0', `%l0')
define(`u1', `%l2')
define(`u2', `%l4')
define(`u3', `%l6')
define(`v0', `%l1')
define(`v1', `%l3')
define(`v2', `%l5')
define(`v3', `%l7')

define(`cy',`%i4')

define(`fanop',`fitod %f0,%f2')		dnl  A quasi nop running in the FA pipe
define(`fmnop',`fmuld %f0,%f0,%f4')	dnl  A quasi nop running in the FM pipe

ASM_START()
	REGISTER(%g2,#scratch)
	REGISTER(%g3,#scratch)
PROLOGUE(mpn_add_nc)
	save	%sp,-160,%sp

	fitod	%f0,%f0		C make sure f0 contains small, quiet number
	subcc	n,4,%g0
	bl,pn	%xcc,.Loop0
	nop
	b,a	L(com)
EPILOGUE()

PROLOGUE(mpn_add_n)
	save	%sp,-160,%sp

	fitod	%f0,%f0		C make sure f0 contains small, quiet number
	subcc	n,4,%g0
	bl,pn	%xcc,.Loop0
	mov	0,cy
L(com):
	ldx	[up+0],u0
	ldx	[vp+0],v0
	add	up,32,up
	ldx	[up-24],u1
	ldx	[vp+8],v1
	add	vp,32,vp
	ldx	[up-16],u2
	ldx	[vp-16],v2
	ldx	[up-8],u3
	ldx	[vp-8],v3
	subcc	n,8,n
	add	u0,v0,%g1	C main add
	add	%g1,cy,%g5	C carry add
	or	u0,v0,%g2
	bl,pn	%xcc,.Lend4567
	fanop
	b,a	.Loop

	.align	16
C START MAIN LOOP
.Loop:	andn	%g2,%g5,%g2
	and	u0,v0,%g3
	ldx	[up+0],u0
	fanop
C --
	or	%g3,%g2,%g2
	ldx	[vp+0],v0
	add	up,32,up
	fanop
C --
	srlx	%g2,63,cy
	add	u1,v1,%g1
	stx	%g5,[rp+0]
	fanop
C --
	add	%g1,cy,%g5
	or	u1,v1,%g2
	fmnop
	fanop
C --
	andn	%g2,%g5,%g2
	and	u1,v1,%g3
	ldx	[up-24],u1
	fanop
C --
	or	%g3,%g2,%g2
	ldx	[vp+8],v1
	add	vp,32,vp
	fanop
C --
	srlx	%g2,63,cy
	add	u2,v2,%g1
	stx	%g5,[rp+8]
	fanop
C --
	add	%g1,cy,%g5
	or	u2,v2,%g2
	fmnop
	fanop
C --
	andn	%g2,%g5,%g2
	and	u2,v2,%g3
	ldx	[up-16],u2
	fanop
C --
	or	%g3,%g2,%g2
	ldx	[vp-16],v2
	add	rp,32,rp
	fanop
C --
	srlx	%g2,63,cy
	add	u3,v3,%g1
	stx	%g5,[rp-16]
	fanop
C --
	add	%g1,cy,%g5
	or	u3,v3,%g2
	fmnop
	fanop
C --
	andn	%g2,%g5,%g2
	and	u3,v3,%g3
	ldx	[up-8],u3
	fanop
C --
	or	%g3,%g2,%g2
	subcc	n,4,n
	ldx	[vp-8],v3
	fanop
C --
	srlx	%g2,63,cy
	add	u0,v0,%g1
	stx	%g5,[rp-8]
	fanop
C --
	add	%g1,cy,%g5
	or	u0,v0,%g2
	bge,pt	%xcc,.Loop
	fanop
C END MAIN LOOP
.Lend4567:
	andn	%g2,%g5,%g2
	and	u0,v0,%g3
	or	%g3,%g2,%g2
	srlx	%g2,63,cy
	add	u1,v1,%g1
	stx	%g5,[rp+0]
	add	%g1,cy,%g5
	or	u1,v1,%g2
	andn	%g2,%g5,%g2
	and	u1,v1,%g3
	or	%g3,%g2,%g2
	srlx	%g2,63,cy
	add	u2,v2,%g1
	stx	%g5,[rp+8]
	add	%g1,cy,%g5
	or	u2,v2,%g2
	andn	%g2,%g5,%g2
	and	u2,v2,%g3
	or	%g3,%g2,%g2
	add	rp,32,rp
	srlx	%g2,63,cy
	add	u3,v3,%g1
	stx	%g5,[rp-16]
	add	%g1,cy,%g5
	or	u3,v3,%g2
	andn	%g2,%g5,%g2
	and	u3,v3,%g3
	or	%g3,%g2,%g2
	srlx	%g2,63,cy
	stx	%g5,[rp-8]

	addcc	n,4,n
	bz,pn	%xcc,.Lret
	fanop

.Loop0:	ldx	[up],u0
	add	up,8,up
	ldx	[vp],v0
	add	vp,8,vp
	add	rp,8,rp
	subcc	n,1,n
	add	u0,v0,%g1
	or	u0,v0,%g2
	add	%g1,cy,%g5
	and	u0,v0,%g3
	andn	%g2,%g5,%g2
	stx	%g5,[rp-8]
	or	%g3,%g2,%g2
	bnz,pt	%xcc,.Loop0
	srlx	%g2,63,cy

.Lret:	mov	cy,%i0
	ret
	restore
EPILOGUE()
