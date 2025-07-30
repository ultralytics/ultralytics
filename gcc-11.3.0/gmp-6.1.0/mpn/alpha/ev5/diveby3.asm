dnl  Alpha mpn_divexact_by3c -- mpn division by 3, expecting no remainder.

dnl  Copyright 2004, 2005, 2009 Free Software Foundation, Inc.

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
C EV4:    22
C EV5:    11.5
C EV6:     6.3		Note that mpn_bdiv_dbm1c is faster

C TODO
C  * Remove the unops, they benefit just ev6, which no longer uses this file.
C  * Try prefetch for destination, using lds.
C  * Improve feed-in code, by moving initial mulq earlier; make initial load
C    to u0/u0 to save some copying.
C  * Combine u0 and u2, u1 and u3.

C INPUT PARAMETERS
define(`rp',	`r16')
define(`up',	`r17')
define(`n',	`r18')
define(`cy',	`r19')

ASM_START()

DATASTART(L(LC),8)
	.quad	0xAAAAAAAAAAAAAAAB
	.quad	0x5555555555555555
	.quad	0xAAAAAAAAAAAAAAAA
DATAEND()

define(`xAAAAAAAAAAAAAAAB',	`r20')
define(`x5555555555555555',	`r21')
define(`xAAAAAAAAAAAAAAAA',	`r22')
define(`u0',	`r0')	define(`u1',	`r1')
define(`u2',	`r2')	define(`u3',	`r3')
define(`l0',	`r25')	define(`x',	`r8')
define(`q0',	`r4')	define(`q1',	`r5')
define(`p6',	`r6')	define(`p7',	`r7')
define(`t0',	`r23')	define(`t1',	`r24')
define(`cymask',`r28')


PROLOGUE(mpn_divexact_by3c,gp)

	ldq	r28, 0(up)			C load first limb early

C Put magic constants in registers
	lda	r0, L(LC)
	ldq	xAAAAAAAAAAAAAAAB, 0(r0)
	ldq	x5555555555555555, 8(r0)
	ldq	xAAAAAAAAAAAAAAAA, 16(r0)

C Compute initial l0 value
	cmpeq	cy, 1, p6
	cmpeq	cy, 2, p7
	negq	p6, p6
	and	p6, x5555555555555555, l0
	cmovne	p7, xAAAAAAAAAAAAAAAA, l0

C Feed-in depending on (n mod 4)
	and	n, 3, r8
	lda	n, -3(n)
	cmpeq	r8, 1, r4
	cmpeq	r8, 2, r5
	bne	r4, $Lb01
	bne	r5, $Lb10
	beq	r8, $Lb00

$Lb11:	ldq	u3, 8(up)
	lda	up, -24(up)
	lda	rp, -24(rp)
	mulq	r28, xAAAAAAAAAAAAAAAB, q0
	mov	r28, u2
	br	r31, $L11

$Lb00:	ldq	u2, 8(up)
	lda	up, -16(up)
	lda	rp, -16(rp)
	mulq	r28, xAAAAAAAAAAAAAAAB, q1
	mov	r28, u1
	br	r31, $L00

$Lb01:	lda	rp, -8(rp)
	mulq	r28, xAAAAAAAAAAAAAAAB, q0
	mov	r28, u0
	blt	n, $Lcj1
	ldq	u1, 8(up)
	lda	up, -8(up)
	br	r31, $L01

$Lb10:	ldq	u0, 8(up)
	mulq	r28, xAAAAAAAAAAAAAAAB, q1
	mov	r28, u3
	blt	n, $Lend

	ALIGN(16)
$Ltop:
C 0
	cmpult	u3, cy, cy			C L0
	mulq	u0, xAAAAAAAAAAAAAAAB, q0	C U1
	ldq	u1, 16(up)			C L1
	addq	q1, l0, x			C U0
C 1
	negq	cy, cymask			C L0
	unop					C U1
	unop					C L1
	cmpult	x5555555555555555, x, p6	C U0
C 2
	cmpult	xAAAAAAAAAAAAAAAA, x, p7	C U1
	unop
	unop
	negq	p6, t0				C L0
C 3
	negq	p7, t1				C L0
	and	cymask, x5555555555555555, l0	C U1
	addq	p6, cy, cy
	and	t0, x5555555555555555, t0
C 4
	and	t1, x5555555555555555, t1
	addq	p7, cy, cy
	unop
	addq	t0, l0, l0
C 5
	addq	t1, l0, l0
	unop
	stq	x, 0(rp)			C L1
	unop
$L01:
C 0
	cmpult	u0, cy, cy			C L0
	mulq	u1, xAAAAAAAAAAAAAAAB, q1	C U1
	ldq	u2, 24(up)			C L1
	addq	q0, l0, x			C U0
C 1
	negq	cy, cymask			C L0
	unop					C U1
	unop					C L1
	cmpult	x5555555555555555, x, p6	C U0
C 2
	cmpult	xAAAAAAAAAAAAAAAA, x, p7	C U1
	unop
	unop
	negq	p6, t0				C L0
C 3
	negq	p7, t1				C L0
	and	cymask, x5555555555555555, l0	C U1
	addq	p6, cy, cy
	and	t0, x5555555555555555, t0
C 4
	and	t1, x5555555555555555, t1
	addq	p7, cy, cy
	unop
	addq	t0, l0, l0
C 5
	addq	t1, l0, l0
	unop
	stq	x, 8(rp)			C L1
	unop
$L00:
C 0
	cmpult	u1, cy, cy			C L0
	mulq	u2, xAAAAAAAAAAAAAAAB, q0	C U1
	ldq	u3, 32(up)			C L1
	addq	q1, l0, x			C U0
C 1
	negq	cy, cymask			C L0
	unop					C U1
	unop					C L1
	cmpult	x5555555555555555, x, p6	C U0
C 2
	cmpult	xAAAAAAAAAAAAAAAA, x, p7	C U1
	unop
	unop
	negq	p6, t0				C L0
C 3
	negq	p7, t1				C L0
	and	cymask, x5555555555555555, l0	C U1
	addq	p6, cy, cy
	and	t0, x5555555555555555, t0
C 4
	and	t1, x5555555555555555, t1
	addq	p7, cy, cy
	unop
	addq	t0, l0, l0
C 5
	addq	t1, l0, l0
	unop
	stq	x, 16(rp)			C L1
	unop
$L11:
C 0
	cmpult	u2, cy, cy			C L0
	mulq	u3, xAAAAAAAAAAAAAAAB, q1	C U1
	ldq	u0, 40(up)			C L1
	addq	q0, l0, x			C U0
C 1
	negq	cy, cymask			C L0
	unop					C U1
	unop					C L1
	cmpult	x5555555555555555, x, p6	C U0
C 2
	cmpult	xAAAAAAAAAAAAAAAA, x, p7	C U1
	lda	n, -4(n)			C L1 bookkeeping
	unop
	negq	p6, t0				C L0
C 3
	negq	p7, t1				C L0
	and	cymask, x5555555555555555, l0	C U1
	addq	p6, cy, cy
	and	t0, x5555555555555555, t0
C 4
	and	t1, x5555555555555555, t1
	addq	p7, cy, cy
	unop
	addq	t0, l0, l0
C 5
	addq	t1, l0, l0
	unop
	stq	x, 24(rp)			C L1
	lda	up, 32(up)
C
	ldl	r31, 256(up)			C prefetch
	unop
	lda	rp, 32(rp)
	bge	n, $Ltop			C U1
C *** MAIN LOOP END ***
$Lend:

	cmpult	u3, cy, cy			C L0
	mulq	u0, xAAAAAAAAAAAAAAAB, q0	C U1
	unop
	addq	q1, l0, x			C U0
C 1
	negq	cy, cymask			C L0
	unop					C U1
	unop					C L1
	cmpult	x5555555555555555, x, p6	C U0
C 2
	cmpult	xAAAAAAAAAAAAAAAA, x, p7	C U1
	unop
	unop
	negq	p6, t0				C L0
C 3
	negq	p7, t1				C L0
	and	cymask, x5555555555555555, l0	C U1
	addq	p6, cy, cy
	and	t0, x5555555555555555, t0
C 4
	and	t1, x5555555555555555, t1
	addq	p7, cy, cy
	unop
	addq	t0, l0, l0
C 5
	addq	t1, l0, l0
	unop
	stq	x, 0(rp)			C L1
	unop
$Lcj1:
	cmpult	u0, cy, cy			C L0
	addq	q0, l0, x			C U0
	cmpult	x5555555555555555, x, p6	C U0
	cmpult	xAAAAAAAAAAAAAAAA, x, p7	C U1
	addq	p6, cy, cy
	addq	p7, cy, r0
	stq	x, 8(rp)			C L1

	ret	r31,(r26),1
EPILOGUE()
ASM_END()

C This is useful for playing with various schedules.
C Expand as: one(0)one(1)one(2)one(3)
define(`one',`
C 0
	cmpult	`$'eval(($1+3)%4), cy, cy		C L0
	mulq	`$'$1, xAAAAAAAAAAAAAAAB, `$'eval(4+$1%2) C U1
	ldq	`$'eval(($1+1)%4), eval($1*8+16)(up)	C L1
	addq	`$'eval(4+($1+1)%2), l0, x		C U0
C 1
	negq	cy, cymask				C L0
	unop						C U1
	unop						C L1
	cmpult	x5555555555555555, x, p6		C U0
C 2
	cmpult	xAAAAAAAAAAAAAAAA, x, p7		C U1
	unop
	unop
	negq	p6, t0					C L0
C 3
	negq	p7, t1					C L0
	and	cymask, x5555555555555555, l0		C U1
	addq	p6, cy, cy
	and	t0, x5555555555555555, t0
C 4
	and	t1, x5555555555555555, t1
	addq	p7, cy, cy
	unop
	addq	t0, l0, l0
C 5
	addq	t1, l0, l0
	unop
	stq	x, eval($1*8)(rp)			C L1
	unop
')
