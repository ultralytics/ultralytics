dnl  Alpha mpn_divrem_2 -- Divide an mpn number by a normalized 2-limb number.

dnl  Copyright 2007, 2008, 2013 Free Software Foundation, Inc.

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

C		norm	frac
C ev4
C ev5		70	70
C ev6		29	29

C TODO
C  * Perhaps inline mpn_invert_limb, that would allow us to not save/restore
C    any registers (thus save ~10 cycles per call).
C  * Use negated d1 and/or d0 to speed carry propagation.  Might save a cycle
C    or two.
C  * Check cluster delays (for ev6).  We very likely could save some cycles.
C  * Use branch-free code for computing di.
C  * CAVEAT: We rely on r19 not being clobbered by mpn_invert_limb call.

C INPUT PARAMETERS
define(`qp',		`r16')
define(`fn',		`r17')
define(`up_param',	`r18')
define(`un_param',	`r19')
define(`dp',		`r20')

ASM_START()
PROLOGUE(mpn_divrem_2,gp)
	lda	r30, -80(r30)
	stq	r26, 0(r30)
	stq	r9, 8(r30)
	stq	r10, 16(r30)
	stq	r11, 24(r30)
	stq	r12, 32(r30)
	stq	r13, 40(r30)
C	stq	r14, 48(r30)
	stq	r15, 56(r30)
	.prologue	1
	stq	r16, 64(r30)
	bis	r31, r17, r15
	s8addq	r19, r18, r13
	lda	r13, -24(r13)
	ldq	r12, 8(r20)
	ldq	r10, 0(r20)
	ldq	r11, 16(r13)
	ldq	r9, 8(r13)

	bis	r31, r31, r3		C most_significant_q_limb = 0
	cmpult	r11, r12, r1
	bne	r1, L(L8)
	cmpule	r11, r12, r1
	cmpult	r9, r10, r2
	and	r1, r2, r1
	bne	r1, L(L8)
	subq	r11, r12, r11
	subq	r11, r2, r11
	subq	r9, r10, r9
	lda	r3, 1(r31)		C most_significant_q_limb = 1
L(L8):	stq	r3, 72(r30)

	addq	r15, r19, r19
	lda	r19, -3(r19)
	blt	r19, L(L10)
	bis	r31, r12, r16
	jsr	r26, mpn_invert_limb
	LDGP(	r29, 0(r26))
	mulq	r0, r12, r4		C t0 = LO(di * d1)
	umulh	r0, r10, r2		C s1 = HI(di * d0)
	addq	r4, r10, r4		C t0 += d0
	cmpule	r10, r4, r7		C (t0 < d0)
	addq	r4, r2, r4		C t0 += s1
	cmpult	r4, r2, r1
	subq	r1, r7, r7		C t1 (-1, 0, or 1)
	blt	r7, L(L42)
L(L22):
	lda	r0, -1(r0)		C di--
	cmpult	r4, r12, r1		C cy for: t0 -= d1 (below)
	subq	r7, r1, r7		C t1 -= cy
	subq	r4, r12, r4		C t0 -= d1
	bge	r7, L(L22)
L(L42):
	ldq	r16, 64(r30)
	s8addq	r19, r16, r16
	ALIGN(16)
L(loop):
	mulq	r11, r0, r5		C q0 (early)
	umulh	r11, r0, r6		C q  (early)
	addq	r5, r9, r8		C q0 += n1
	addq	r6, r11, r6		C q  += n2
	cmpult	r8, r5, r1		C cy for: q0 += n1
	addq	r6, r1, r6		C q  += cy
	unop
	mulq	r12, r6, r1		C LO(d1 * q)
	umulh	r10, r6, r7		C t1 = HI(d0 * q)
	subq	r9, r1, r9		C n1 -= LO(d1 * q)
	mulq	r10, r6, r4		C t0 = LO(d0 * q)
	unop
	cmple	r15, r19, r5		C condition and n0...
	beq	r5, L(L31)
	ldq	r5, 0(r13)
	lda	r13, -8(r13)
L(L31):	subq	r9, r12, r9		C n1 -= d1
	cmpult	r5, r10, r1		C
	subq	r9, r1, r9		C
	subq	r5, r10, r5		C n0 -= d0
	subq	r9, r7, r9		C n1 -= t0
	cmpult	r5, r4, r1		C
	subq	r9, r1, r2		C
	subq	r5, r4, r5		C n0 -= t1
	cmpult	r2, r8, r1		C (n1 < q0)
	addq	r6, r1, r6		C q += cond
	lda	r1, -1(r1)		C -(n1 >= q0)
	and	r1, r10, r4		C
	addq	r5, r4, r9		C n0 += mask & d0
	and	r1, r12, r1		C
	cmpult	r9, r5, r11		C cy for: n0 += mask & d0
	addq	r2, r1, r1		C n1 += mask & d1
	addq	r1, r11, r11		C n1 += cy
	cmpult	r11, r12, r1		C
	beq	r1, L(fix)		C
L(bck):	stq	r6, 0(r16)
	lda	r16, -8(r16)
	lda	r19, -1(r19)
	bge	r19, L(loop)

L(L10):	stq	r9, 8(r13)
	stq	r11, 16(r13)
	ldq	r0, 72(r30)
	ldq	r26, 0(r30)
	ldq	r9, 8(r30)
	ldq	r10, 16(r30)
	ldq	r11, 24(r30)
	ldq	r12, 32(r30)
	ldq	r13, 40(r30)
C	ldq	r14, 48(r30)
	ldq	r15, 56(r30)
	lda	r30, 80(r30)
	ret	r31, (r26), 1

L(fix):	cmpule	r11, r12, r1
	cmpult	r9, r10, r2
	and	r1, r2, r1
	bne	r1, L(bck)
	subq	r11, r12, r11
	subq	r11, r2, r11
	subq	r9, r10, r9
	lda	r6, 1(r6)
	br	L(bck)
EPILOGUE()
ASM_END()
