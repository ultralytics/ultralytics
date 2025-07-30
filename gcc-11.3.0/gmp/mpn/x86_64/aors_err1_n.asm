dnl  AMD64 mpn_add_err1_n, mpn_sub_err1_n

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
C AMD K8,K9	 2.75 (degenerates to 3 c/l for some alignments)
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
define(`yp',	`%r8')
define(`n',	`%r9')
define(`cy_param',	`8(%rsp)')

define(`el',	`%rbx')
define(`eh',	`%rbp')
define(`t0',	`%r10')
define(`t1',	`%r11')
define(`t2',	`%r12')
define(`t3',	`%r13')
define(`w0',	`%r14')
define(`w1',	`%r15')

ifdef(`OPERATION_add_err1_n', `
	define(ADCSBB,	      adc)
	define(func,	      mpn_add_err1_n)')
ifdef(`OPERATION_sub_err1_n', `
	define(ADCSBB,	      sbb)
	define(func,	      mpn_sub_err1_n)')

MULFUNC_PROLOGUE(mpn_add_err1_n mpn_sub_err1_n)


ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(func)
	mov	cy_param, %rax

	push	%rbx
	push	%rbp
	push	%r12
	push	%r13
	push	%r14
	push	%r15

	lea	(up,n,8), up
	lea	(vp,n,8), vp
	lea	(rp,n,8), rp

	mov	R32(n), R32(%r10)
	and	$3, R32(%r10)
	jz	L(0mod4)
	cmp	$2, R32(%r10)
	jc	L(1mod4)
	jz	L(2mod4)
L(3mod4):
	xor	R32(el), R32(el)
	xor	R32(eh), R32(eh)
	xor	R32(t0), R32(t0)
	xor	R32(t1), R32(t1)
	lea	-24(yp,n,8), yp
	neg	n

	shr	$1, %al		   C restore carry
	mov	(up,n,8), w0
	mov	8(up,n,8), w1
	ADCSBB	(vp,n,8), w0
	mov	w0, (rp,n,8)
	cmovc	16(yp), el
	ADCSBB	8(vp,n,8), w1
	mov	w1, 8(rp,n,8)
	cmovc	8(yp), t0
	mov	16(up,n,8), w0
	ADCSBB	16(vp,n,8), w0
	mov	w0, 16(rp,n,8)
	cmovc	(yp), t1
	setc	%al		   C save carry
	add	t0, el
	adc	$0, eh
	add	t1, el
	adc	$0, eh

	add	$3, n
	jnz	L(loop)
	jmp	L(end)

	ALIGN(16)
L(0mod4):
	xor	R32(el), R32(el)
	xor	R32(eh), R32(eh)
	lea	(yp,n,8), yp
	neg	n
	jmp	L(loop)

	ALIGN(16)
L(1mod4):
	xor	R32(el), R32(el)
	xor	R32(eh), R32(eh)
	lea	-8(yp,n,8), yp
	neg	n

	shr	$1, %al		   C restore carry
	mov	(up,n,8), w0
	ADCSBB	(vp,n,8), w0
	mov	w0, (rp,n,8)
	cmovc	(yp), el
	setc	%al		   C save carry

	add	$1, n
	jnz	L(loop)
	jmp	L(end)

	ALIGN(16)
L(2mod4):
	xor	R32(el), R32(el)
	xor	R32(eh), R32(eh)
	xor	R32(t0), R32(t0)
	lea	-16(yp,n,8), yp
	neg	n

	shr	$1, %al		   C restore carry
	mov	(up,n,8), w0
	mov	8(up,n,8), w1
	ADCSBB	(vp,n,8), w0
	mov	w0, (rp,n,8)
	cmovc	8(yp), el
	ADCSBB	8(vp,n,8), w1
	mov	w1, 8(rp,n,8)
	cmovc	(yp), t0
	setc	%al		   C save carry
	add	t0, el
	adc	$0, eh

	add	$2, n
	jnz	L(loop)
	jmp	L(end)

	ALIGN(32)
L(loop):
	shr	$1, %al		   C restore carry
	mov	-8(yp), t0
	mov	$0, R32(t3)
	mov	(up,n,8), w0
	mov	8(up,n,8), w1
	ADCSBB	(vp,n,8), w0
	cmovnc	t3, t0
	ADCSBB	8(vp,n,8), w1
	mov	-16(yp), t1
	mov	w0, (rp,n,8)
	mov	16(up,n,8), w0
	mov	w1, 8(rp,n,8)
	cmovnc	t3, t1
	mov	-24(yp), t2
	ADCSBB	16(vp,n,8), w0
	cmovnc	t3, t2
	mov	24(up,n,8), w1
	ADCSBB	24(vp,n,8), w1
	cmovc	-32(yp), t3
	setc	%al		   C save carry
	add	t0, el
	adc	$0, eh
	add	t1, el
	adc	$0, eh
	add	t2, el
	adc	$0, eh
	mov	w0, 16(rp,n,8)
	add	t3, el
	lea	-32(yp), yp
	adc	$0, eh
	mov	w1, 24(rp,n,8)
	add	$4, n
	jnz	L(loop)

L(end):
	mov	el, (ep)
	mov	eh, 8(ep)

	pop	%r15
	pop	%r14
	pop	%r13
	pop	%r12
	pop	%rbp
	pop	%rbx
	ret
EPILOGUE()
