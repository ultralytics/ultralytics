dnl  AMD64 mpn_add_err2_n, mpn_sub_err2_n

dnl  Contributed by David Harvey.

dnl  Copyright 2011 Free Software Foundation, Inc.

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

C	     cycles/limb
C AMD K8,K9	 4.5
C AMD K10	 ?
C Intel P4	 ?
C Intel core2	 6.9
C Intel corei	 ?
C Intel atom	 ?
C VIA nano	 ?


C INPUT PARAMETERS
define(`rp',	`%rdi')
define(`up',	`%rsi')
define(`vp',	`%rdx')
define(`ep',	`%rcx')
define(`yp1',	`%r8')
define(`yp2',   `%r9')
define(`n_param',     `8(%rsp)')
define(`cy_param',    `16(%rsp)')

define(`cy1',   `%r14')
define(`cy2',   `%rax')

define(`n',     `%r10')

define(`w',     `%rbx')
define(`e1l',	`%rbp')
define(`e1h',	`%r11')
define(`e2l',	`%r12')
define(`e2h',	`%r13')


ifdef(`OPERATION_add_err2_n', `
	define(ADCSBB,	      adc)
	define(func,	      mpn_add_err2_n)')
ifdef(`OPERATION_sub_err2_n', `
	define(ADCSBB,	      sbb)
	define(func,	      mpn_sub_err2_n)')

MULFUNC_PROLOGUE(mpn_add_err2_n mpn_sub_err2_n)


ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(func)
	mov	cy_param, cy2
	mov	n_param, n

	push	%rbx
	push	%rbp
	push	%r12
	push	%r13
	push	%r14

	xor	R32(e1l), R32(e1l)
	xor	R32(e1h), R32(e1h)
	xor	R32(e2l), R32(e2l)
	xor	R32(e2h), R32(e2h)

	sub	yp1, yp2

	lea	(rp,n,8), rp
	lea	(up,n,8), up
	lea	(vp,n,8), vp

	test	$1, n
	jnz	L(odd)

	lea	-8(yp1,n,8), yp1
	neg	n
	jmp	L(top)

	ALIGN(16)
L(odd):
	lea	-16(yp1,n,8), yp1
	neg	n
	shr	$1, cy2
	mov	(up,n,8), w
	ADCSBB	(vp,n,8), w
	cmovc	8(yp1), e1l
	cmovc	8(yp1,yp2), e2l
	mov	w, (rp,n,8)
	sbb	cy2, cy2
	inc	n
	jz	L(end)

	ALIGN(16)
L(top):
	mov	(up,n,8), w
	shr	$1, cy2		C restore carry
	ADCSBB	(vp,n,8), w
	mov	w, (rp,n,8)
	sbb	cy1, cy1	C generate mask, preserve CF

	mov	8(up,n,8), w
	ADCSBB	8(vp,n,8), w
	mov	w, 8(rp,n,8)
	sbb	cy2, cy2	C generate mask, preserve CF

	mov	(yp1), w	C (e1h:e1l) += cy1 * yp1 limb
	and	cy1, w
	add	w, e1l
	adc	$0, e1h

	and	(yp1,yp2), cy1	C (e2h:e2l) += cy1 * yp2 limb
	add	cy1, e2l
	adc	$0, e2h

	mov	-8(yp1), w	C (e1h:e1l) += cy2 * next yp1 limb
	and	cy2, w
	add	w, e1l
	adc	$0, e1h

	mov	-8(yp1,yp2), w	C (e2h:e2l) += cy2 * next yp2 limb
	and	cy2, w
	add	w, e2l
	adc	$0, e2h

	add	$2, n
	lea	-16(yp1), yp1
	jnz	L(top)
L(end):

	mov	e1l, (ep)
	mov	e1h, 8(ep)
	mov	e2l, 16(ep)
	mov	e2h, 24(ep)

	and	$1, %eax	C return carry

	pop	%r14
	pop	%r13
	pop	%r12
	pop	%rbp
	pop	%rbx
	ret
EPILOGUE()
