dnl  Alpha mpn_addlsh1_n/mpn_sublsh1_n -- rp[] = up[] +- (vp[] << 1).

dnl  Copyright 2003, 2013 Free Software Foundation, Inc.

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

C      cycles/limb
C EV4:     ?
C EV5:     6.25
C EV6:     4.5

define(`rp',`r16')
define(`up',`r17')
define(`vp',`r18')
define(`n', `r19')

define(`u0', `r8')
define(`u1', `r1')
define(`v0', `r4')
define(`v1', `r5')

define(`cy0', `r0')
define(`cy1', `r20')
define(`cy', `r22')
define(`rr', `r24')
define(`ps', `r25')
define(`sl', `r28')

ifdef(`OPERATION_addlsh1_n',`
  define(ADDSUB,       addq)
  define(CARRY,       `cmpult $1,$2,$3')
  define(func, mpn_addlsh1_n)
')
ifdef(`OPERATION_sublsh1_n',`
  define(ADDSUB,       subq)
  define(CARRY,       `cmpult $2,$1,$3')
  define(func, mpn_sublsh1_n)
')

MULFUNC_PROLOGUE(mpn_addlsh1_n mpn_sublsh1_n)

ASM_START()
PROLOGUE(func)
	and	n, 2, cy0
	blbs	n, L(bx1)
L(bx0):	ldq	v1, 0(vp)
	ldq	u1, 0(up)
	nop
	bne	cy0, L(b10)

L(b00):	lda	vp, 48(vp)
	lda	up, -16(up)
	lda	rp, -8(rp)
	br	r31, L(lo0)

L(b10):	lda	vp, 32(vp)
	lda	rp, 8(rp)
	lda	cy0, 0(r31)
	br	r31, L(lo2)

L(bx1):	ldq	v0, 0(vp)
	ldq	u0, 0(up)
	lda	cy1, 0(r31)
	beq	cy0, L(b01)

L(b11):	lda	vp, 40(vp)
	lda	up, -24(up)
	lda	rp, 16(rp)
	br	r31, L(lo3)

L(b01):	lda	n, -4(n)
	ble	n, L(end)
	lda	vp, 24(vp)
	lda	up, -8(up)

	ALIGN(16)
L(top):	addq	v0, v0, sl	C left shift vlimb
	ldq	v1, -16(vp)
	ADDSUB	u0, sl, ps	C ulimb + (vlimb << 1)
	cmplt	v0, r31, cy0	C carry out #1
	ldq	u1, 16(up)
	ADDSUB	ps, cy1, rr	C consume carry from previous operation
	CARRY(	ps, u0, cy)	C carry out #2
	stq	rr, 0(rp)
	addq	cy, cy0, cy0	C combine carry out #1 and #2
	CARRY(	rr, ps, cy)	C carry out #3
	addq	cy, cy0, cy0	C final carry out
	lda	vp, 32(vp)	C bookkeeping
L(lo0):	addq	v1, v1, sl
	ldq	v0, -40(vp)
	ADDSUB	u1, sl, ps
	cmplt	v1, r31, cy1
	ldq	u0, 24(up)
	ADDSUB	ps, cy0, rr
	CARRY(	ps, u1, cy)
	stq	rr, 8(rp)
	addq	cy, cy1, cy1
	CARRY(	rr, ps, cy)
	addq	cy, cy1, cy1
	lda	rp, 32(rp)	C bookkeeping
L(lo3):	addq	v0, v0, sl
	ldq	v1, -32(vp)
	ADDSUB	u0, sl, ps
	cmplt	v0, r31, cy0
	ldq	u1, 32(up)
	ADDSUB	ps, cy1, rr
	CARRY(	ps, u0, cy)
	stq	rr, -16(rp)
	addq	cy, cy0, cy0
	CARRY(	rr, ps, cy)
	addq	cy, cy0, cy0
	lda	up, 32(up)	C bookkeeping
L(lo2):	addq	v1, v1, sl
	ldq	v0, -24(vp)
	ADDSUB	u1, sl, ps
	cmplt	v1, r31, cy1
	ldq	u0, 8(up)
	ADDSUB	ps, cy0, rr
	CARRY(	ps, u1, cy)
	stq	rr, -8(rp)
	addq	cy, cy1, cy1
	CARRY(	rr, ps, cy)
	addq	cy, cy1, cy1
	lda	n, -4(n)	C bookkeeping
	bgt	n, L(top)

L(end):	addq	v0, v0, sl
	ADDSUB	u0, sl, ps
	ADDSUB	ps, cy1, rr
	cmplt	v0, r31, cy0
	CARRY(	ps, u0, cy)
	stq	rr, 0(rp)
	addq	cy, cy0, cy0
	CARRY(	rr, ps, cy)
	addq	cy, cy0, r0
	ret	r31,(r26),1
EPILOGUE()
ASM_END()
