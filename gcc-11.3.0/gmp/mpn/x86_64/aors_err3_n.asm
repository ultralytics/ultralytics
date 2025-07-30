dnl  AMD64 mpn_add_err3_n, mpn_sub_err3_n

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
C AMD K8,K9	 7.0
C AMD K10	 ?
C Intel P4	 ?
C Intel core2	 ?
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
define(`yp3_param',   `8(%rsp)')
define(`n_param',     `16(%rsp)')
define(`cy_param',    `24(%rsp)')

define(`n',     `%r10')
define(`yp3',   `%rcx')
define(`t',     `%rbx')

define(`e1l',	`%rbp')
define(`e1h',	`%r11')
define(`e2l',	`%r12')
define(`e2h',	`%r13')
define(`e3l',   `%r14')
define(`e3h',   `%r15')



ifdef(`OPERATION_add_err3_n', `
	define(ADCSBB,	      adc)
	define(func,	      mpn_add_err3_n)')
ifdef(`OPERATION_sub_err3_n', `
	define(ADCSBB,	      sbb)
	define(func,	      mpn_sub_err3_n)')

MULFUNC_PROLOGUE(mpn_add_err3_n mpn_sub_err3_n)


ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(func)
	mov	cy_param, %rax
	mov	n_param, n

	push	%rbx
	push	%rbp
	push	%r12
	push	%r13
	push	%r14
	push	%r15

	push	ep
	mov	64(%rsp), yp3       C load from yp3_param

	xor	R32(e1l), R32(e1l)
	xor	R32(e1h), R32(e1h)
	xor	R32(e2l), R32(e2l)
	xor	R32(e2h), R32(e2h)
	xor	R32(e3l), R32(e3l)
	xor	R32(e3h), R32(e3h)

	sub	yp1, yp2
	sub	yp1, yp3

	lea	-8(yp1,n,8), yp1
	lea	(rp,n,8), rp
	lea	(up,n,8), up
	lea	(vp,n,8), vp
	neg	n

	ALIGN(16)
L(top):
	shr	$1, %rax		C restore carry
	mov	(up,n,8), %rax
	ADCSBB	(vp,n,8), %rax
	mov	%rax, (rp,n,8)
	sbb	%rax, %rax		C save carry and generate mask

	mov	(yp1), t
	and	%rax, t
	add	t, e1l
	adc	$0, e1h

	mov	(yp1,yp2), t
	and	%rax, t
	add	t, e2l
	adc	$0, e2h

	mov	(yp1,yp3), t
	and	%rax, t
	add	t, e3l
	adc	$0, e3h

	lea	-8(yp1), yp1
	inc	n
	jnz     L(top)

L(end):
	and	$1, %eax
	pop	ep

	mov	e1l, (ep)
	mov	e1h, 8(ep)
	mov	e2l, 16(ep)
	mov	e2h, 24(ep)
	mov	e3l, 32(ep)
	mov	e3h, 40(ep)

	pop	%r15
	pop	%r14
	pop	%r13
	pop	%r12
	pop	%rbp
	pop	%rbx
	ret
EPILOGUE()
