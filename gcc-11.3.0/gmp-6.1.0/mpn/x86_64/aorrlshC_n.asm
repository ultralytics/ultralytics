dnl  AMD64 mpn_addlshC_n -- rp[] = up[] + (vp[] << C)
dnl  AMD64 mpn_rsblshC_n -- rp[] = (vp[] << C) - up[]

dnl  Copyright 2009-2012 Free Software Foundation, Inc.

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


C	     cycles/limb
C AMD K8,K9	 2
C AMD K10	 2
C Intel P4	 ?
C Intel core2	 3
C Intel NHM	 2.75
C Intel SBR	 2.55
C Intel atom	 ?
C VIA nano	 ?

C INPUT PARAMETERS
define(`rp',	`%rdi')
define(`up',	`%rsi')
define(`vp',	`%rdx')
define(`n',	`%rcx')

define(M, eval(m4_lshift(1,LSH)))

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(func)
	FUNC_ENTRY(4)
	push	%r12
	push	%r13
	push	%r14
	push	%r15

	mov	(vp), %r8
	lea	(,%r8,M), %r12
	shr	$RSH, %r8

	mov	R32(n), R32(%rax)
	lea	(rp,n,8), rp
	lea	(up,n,8), up
	lea	(vp,n,8), vp
	neg	n
	and	$3, R8(%rax)
	je	L(b00)
	cmp	$2, R8(%rax)
	jc	L(b01)
	je	L(b10)

L(b11):	mov	8(vp,n,8), %r10
	lea	(%r8,%r10,M), %r14
	shr	$RSH, %r10
	mov	16(vp,n,8), %r11
	lea	(%r10,%r11,M), %r15
	shr	$RSH, %r11
	ADDSUB	(up,n,8), %r12
	ADCSBB	8(up,n,8), %r14
	ADCSBB	16(up,n,8), %r15
	sbb	R32(%rax), R32(%rax)		  C save carry for next
	mov	%r12, (rp,n,8)
	mov	%r14, 8(rp,n,8)
	mov	%r15, 16(rp,n,8)
	add	$3, n
	js	L(top)
	jmp	L(end)

L(b01):	mov	%r8, %r11
	ADDSUB	(up,n,8), %r12
	sbb	R32(%rax), R32(%rax)		  C save carry for next
	mov	%r12, (rp,n,8)
	add	$1, n
	js	L(top)
	jmp	L(end)

L(b10):	mov	8(vp,n,8), %r11
	lea	(%r8,%r11,M), %r15
	shr	$RSH, %r11
	ADDSUB	(up,n,8), %r12
	ADCSBB	8(up,n,8), %r15
	sbb	R32(%rax), R32(%rax)		  C save carry for next
	mov	%r12, (rp,n,8)
	mov	%r15, 8(rp,n,8)
	add	$2, n
	js	L(top)
	jmp	L(end)

L(b00):	mov	8(vp,n,8), %r9
	mov	16(vp,n,8), %r10
	jmp	L(e00)

	ALIGN(16)
L(top):	mov	16(vp,n,8), %r10
	mov	(vp,n,8), %r8
	mov	8(vp,n,8), %r9
	lea	(%r11,%r8,M), %r12
	shr	$RSH, %r8
L(e00):	lea	(%r8,%r9,M), %r13
	shr	$RSH, %r9
	mov	24(vp,n,8), %r11
	lea	(%r9,%r10,M), %r14
	shr	$RSH, %r10
	lea	(%r10,%r11,M), %r15
	shr	$RSH, %r11
	add	R32(%rax), R32(%rax)		  C restore carry
	ADCSBB	(up,n,8), %r12
	ADCSBB	8(up,n,8), %r13
	ADCSBB	16(up,n,8), %r14
	ADCSBB	24(up,n,8), %r15
	mov	%r12, (rp,n,8)
	mov	%r13, 8(rp,n,8)
	mov	%r14, 16(rp,n,8)
	sbb	R32(%rax), R32(%rax)		  C save carry for next
	mov	%r15, 24(rp,n,8)
	add	$4, n
	js	L(top)
L(end):

ifelse(ADDSUB,add,`
	sub	R32(%r11), R32(%rax)
	neg	R32(%rax)
',`
	add	R32(%r11), R32(%rax)
	movslq	R32(%rax), %rax
')
	pop	%r15
	pop	%r14
	pop	%r13
	pop	%r12
	FUNC_EXIT()
	ret
EPILOGUE()
