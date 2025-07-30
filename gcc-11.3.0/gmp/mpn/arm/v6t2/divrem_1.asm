dnl  ARM v6t2 mpn_divrem_1 and mpn_preinv_divrem_1.

dnl  Contributed to the GNU project by Torbjörn Granlund.

dnl  Copyright 2012 Free Software Foundation, Inc.

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

C		norm	unorm	frac
C StrongARM	 -	 -	 -
C XScale	 -	 -	 -
C Cortex-A7	 ?	 ?	 ?
C Cortex-A8	 ?	 ?	 ?
C Cortex-A9	13	14	13
C Cortex-A15	11.4	11.8	11.1

C TODO
C  * Optimise inner-loops better, they could likely run a cycle or two faster.
C  * Decrease register usage, streamline non-loop code.

define(`qp_arg',  `r0')
define(`fn',      `r1')
define(`up_arg',  `r2')
define(`n_arg',   `r3')
define(`d_arg',   `0')
define(`dinv_arg',`4')
define(`cnt_arg', `8')

define(`n',       `r9')
define(`qp',      `r5')
define(`up',      `r6')
define(`cnt',     `r7')
define(`tnc',     `r10')
define(`dinv',    `r0')
define(`d',       `r4')

ASM_START()
PROLOGUE(mpn_preinv_divrem_1)
	stmfd	sp!, {r4, r5, r6, r7, r8, r9, r10, r11, lr}
	ldr	d,    [sp, #9*4+d_arg]
	ldr	cnt,  [sp, #9*4+cnt_arg]
	str	r1, [sp, #9*4+d_arg]	C reuse d stack slot for fn
	sub	n, r3, #1
	add	r3, r1, n
	cmp	d, #0
	add	qp, qp_arg, r3, lsl #2	C put qp at Q[] end
	add	up, up_arg, n, lsl #2	C put up at U[] end
	ldr	dinv, [sp, #9*4+dinv_arg]
	blt	L(nent)
	b	L(uent)
EPILOGUE()

PROLOGUE(mpn_divrem_1)
	stmfd	sp!, {r4, r5, r6, r7, r8, r9, r10, r11, lr}
	sub	n, r3, #1
	ldr	d, [sp, #9*4+d_arg]	C d
	str	r1, [sp, #9*4+d_arg]	C reuse d stack slot for fn
	add	r3, r1, n
	cmp	d, #0
	add	qp, qp_arg, r3, lsl #2	C put qp at Q[] end
	add	up, up_arg, n, lsl #2	C put up at U[] end
	blt	L(normalised)

L(unnorm):
	clz	cnt, d
	mov	r0, d, lsl cnt		C pass d << cnt
	bl	mpn_invert_limb
L(uent):
	mov	d, d, lsl cnt		C d <<= cnt
	cmp	n, #0
	mov	r1, #0			C r
	blt	L(frac)

	ldr	r11, [up, #0]

	rsb	tnc, cnt, #32
	mov	r1, r11, lsr tnc
	mov	r11, r11, lsl cnt
	beq	L(uend)

	ldr	r3, [up, #-4]!
	orr	r2, r11, r3, lsr tnc
	b	L(mid)

L(utop):
	mls	r1, d, r8, r11
	mov	r11, r3, lsl cnt
	ldr	r3, [up, #-4]!
	cmp	r1, r2
	addhi	r1, r1, d
	subhi	r8, r8, #1
	orr	r2, r11, r3, lsr tnc
	cmp	r1, d
	bcs	L(ufx)
L(uok):	str	r8, [qp], #-4
L(mid):	add	r8, r1, #1
	mov	r11, r2
	umlal	r2, r8, r1, dinv
	subs	n, n, #1
	bne	L(utop)

	mls	r1, d, r8, r11
	mov	r11, r3, lsl cnt
	cmp	r1, r2
	addhi	r1, r1, d
	subhi	r8, r8, #1
	cmp	r1, d
	rsbcs	r1, d, r1
	addcs	r8, r8, #1
	str	r8, [qp], #-4

L(uend):add	r8, r1, #1
	mov	r2, r11
	umlal	r2, r8, r1, dinv
	mls	r1, d, r8, r11
	cmp	r1, r2
	addhi	r1, r1, d
	subhi	r8, r8, #1
	cmp	r1, d
	rsbcs	r1, d, r1
	addcs	r8, r8, #1
	str	r8, [qp], #-4
L(frac):
	ldr	r2, [sp, #9*4+d_arg]	C fn
	cmp	r2, #0
	beq	L(fend)

L(ftop):mov	r6, #0
	add	r3, r1, #1
	umlal	r6, r3, r1, dinv
	mov	r8, #0
	mls	r1, d, r3, r8
	cmp	r1, r6
	addhi	r1, r1, d
	subhi	r3, r3, #1
	subs	r2, r2, #1
	str	r3, [qp], #-4
	bne	L(ftop)

L(fend):mov	r11, r1, lsr cnt
L(rtn):	mov	r0, r11
	ldmfd	sp!, {r4, r5, r6, r7, r8, r9, r10, r11, pc}

L(normalised):
	mov	r0, d
	bl	mpn_invert_limb
L(nent):
	cmp	n, #0
	mov	r11, #0			C r
	blt	L(nend)

	ldr	r11, [up, #0]
	cmp	r11, d
	movlo	r2, #0			C hi q limb
	movhs	r2, #1			C hi q limb
	subhs	r11, r11, d

	str	r2, [qp], #-4
	cmp	n, #0
	beq	L(nend)

L(ntop):ldr	r1, [up, #-4]!
	add	r12, r11, #1
	umlal	r1, r12, r11, dinv
	ldr	r3, [up, #0]
	mls	r11, d, r12, r3
	cmp	r11, r1
	addhi	r11, r11, d
	subhi	r12, r12, #1
	cmp	d, r11
	bls	L(nfx)
L(nok):	str	r12, [qp], #-4
	subs	n, n, #1
	bne	L(ntop)

L(nend):mov	r1, r11			C r
	mov	cnt, #0			C shift cnt
	b	L(frac)

L(nfx):	add	r12, r12, #1
	rsb	r11, d, r11
	b	L(nok)
L(ufx):	rsb	r1, d, r1
	add	r8, r8, #1
	b	L(uok)
EPILOGUE()
