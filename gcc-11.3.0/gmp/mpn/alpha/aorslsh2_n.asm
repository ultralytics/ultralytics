dnl  Alpha mpn_addlsh2_n/mpn_sublsh2_n -- rp[] = up[] +- (vp[] << 2).

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
C EV5:     6
C EV6:     3.75

C TODO
C  * Tune to reach 3.5 c/l on ev6 and 5.75 c/l on ev5.

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

ifdef(`OPERATION_addlsh2_n',`
  define(ADDSUB,       addq)
  define(CARRY,       `cmpult $1,$2,$3')
  define(func, mpn_addlsh2_n)
')
ifdef(`OPERATION_sublsh2_n',`
  define(ADDSUB,       subq)
  define(CARRY,       `cmpult $2,$1,$3')
  define(func, mpn_sublsh2_n)
')

MULFUNC_PROLOGUE(mpn_addlsh2_n mpn_sublsh2_n)

ASM_START()
PROLOGUE(func)
	and	n, 2, cy0
	blbs	n, L(bx1)
L(bx0):	ldq	v1, 0(vp)
	ldq	u1, 0(up)
	bis	r31, r31, r2
	bne	cy0, L(b10)

L(b00):	lda	vp, 48(vp)
	lda	up, -16(up)
	lda	rp, -8(rp)
	s4addq	v1, r31, sl
	br	r31, L(lo0)

L(b10):	lda	vp, 32(vp)
	lda	rp, 8(rp)
	lda	cy0, 0(r31)
	br	r31, L(lo2)

L(bx1):	ldq	v0, 0(vp)
	ldq	u0, 0(up)
	lda	cy1, 0(r31)
	bis	r31, r31, r3
	nop
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
L(top):	s4addq	v0, r3, sl	C combined vlimb
	ldq	v1, -16(vp)
	ADDSUB	u0, sl, ps	C ulimb + (vlimb << 1)
	ldq	u1, 16(up)
	srl	v0, 62, r2	C high v bits
	ADDSUB	ps, cy1, rr	C consume carry from previous operation
	CARRY(	ps, u0, cy0)	C carry out #2
	stq	rr, 0(rp)
	CARRY(	rr, ps, cy)	C carry out #3
	lda	vp, 32(vp)	C bookkeeping
	addq	cy, cy0, cy0	C final carry out
	s4addq	v1, r2, sl
L(lo0):	ldq	v0, -40(vp)
	ADDSUB	u1, sl, ps
	ldq	u0, 24(up)
	srl	v1, 62, r3
	ADDSUB	ps, cy0, rr
	CARRY(	ps, u1, cy1)
	stq	rr, 8(rp)
	CARRY(	rr, ps, cy)
	lda	rp, 32(rp)	C bookkeeping
	addq	cy, cy1, cy1
L(lo3):	s4addq	v0, r3, sl
	ldq	v1, -32(vp)
	ADDSUB	u0, sl, ps
	ldq	u1, 32(up)
	srl	v0, 62, r2
	ADDSUB	ps, cy1, rr
	CARRY(	ps, u0, cy0)
	stq	rr, -16(rp)
	CARRY(	rr, ps, cy)
	lda	up, 32(up)	C bookkeeping
	addq	cy, cy0, cy0
L(lo2):	s4addq	v1, r2, sl
	ldq	v0, -24(vp)
	ADDSUB	u1, sl, ps
	ldq	u0, 8(up)
	srl	v1, 62, r3
	ADDSUB	ps, cy0, rr
	CARRY(	ps, u1, cy1)
	stq	rr, -8(rp)
	CARRY(	rr, ps, cy)
	lda	n, -4(n)	C bookkeeping
	addq	cy, cy1, cy1
	bgt	n, L(top)

L(end):	s4addq	v0, r3, sl
	ADDSUB	u0, sl, ps
	srl	v0, 62, r2
	ADDSUB	ps, cy1, rr
	CARRY(	ps, u0, cy0)
	stq	rr, 0(rp)
	CARRY(	rr, ps, cy)
	addq	cy, cy0, cy0
	addq	cy0, r2, r0

	ret	r31,(r26),1
EPILOGUE()
ASM_END()
