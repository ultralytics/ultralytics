dnl  ARM mpn_udiv_qrnnd -- divide a two limb dividend and a one limb divisor.
dnl  Return quotient and store remainder through a supplied pointer.

dnl  Copyright 2001, 2012 Free Software Foundation, Inc.

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

C INPUT PARAMETERS
define(`rem_ptr',`r0')
define(`n1',`r1')
define(`n0',`r2')
define(`d',`r3')

C divstep -- develop one quotient bit.  Dividend in $1$2, divisor in $3.
C Quotient bit is shifted into $2.
define(`divstep',
       `adcs	$2, $2, $2
	adc	$1, $1, $1
	cmp	$1, $3
	subcs	$1, $1, $3')

ASM_START()
PROLOGUE(mpn_udiv_qrnnd)
	mov	r12, #8			C loop counter for both loops below
	cmp	d, #0x80000000		C check divisor msb and clear carry
	bcs	L(_large_divisor)

L(oop):	divstep(n1,n0,d)
	divstep(n1,n0,d)
	divstep(n1,n0,d)
	divstep(n1,n0,d)
	sub	r12, r12, #1
	teq	r12, #0
	bne	L(oop)

	str	n1, [rem_ptr]		C store remainder
	adc	r0, n0, n0		C quotient: add last carry from divstep
	bx	lr

L(_large_divisor):
	stmfd	sp!, { r8, lr }

	and	r8, n0, #1		C save lsb of dividend
	mov	lr, n1, lsl #31
	orrs	n0, lr, n0, lsr #1	C n0 = lo(n1n0 >> 1)
	mov	n1, n1, lsr #1		C n1 = hi(n1n0 >> 1)

	and	lr, d, #1		C save lsb of divisor
	movs	d, d, lsr #1		C d = floor(orig_d / 2)
	adc	d, d, #0		C d = ceil(orig_d / 2)

L(oop2):
	divstep(n1,n0,d)
	divstep(n1,n0,d)
	divstep(n1,n0,d)
	divstep(n1,n0,d)
	sub	r12, r12, #1
	teq	r12, #0
	bne	L(oop2)

	adc	n0, n0, n0		C shift and add last carry from divstep
	add	n1, r8, n1, lsl #1	C shift in omitted dividend lsb
	tst	lr, lr			C test saved divisor lsb
	beq	L(_even_divisor)

	rsb	d, lr, d, lsl #1	C restore orig d value
	adds	n1, n1, n0		C fix remainder for omitted divisor lsb
	addcs	n0, n0, #1		C adjust quotient if rem. fix carried
	subcs	n1, n1, d		C adjust remainder accordingly
	cmp	n1, d			C remainder >= divisor?
	subcs	n1, n1, d		C adjust remainder
	addcs	n0, n0, #1		C adjust quotient

L(_even_divisor):
	str	n1, [rem_ptr]		C store remainder
	mov	r0, n0			C quotient
	ldmfd	sp!, { r8, pc }
EPILOGUE(mpn_udiv_qrnnd)
