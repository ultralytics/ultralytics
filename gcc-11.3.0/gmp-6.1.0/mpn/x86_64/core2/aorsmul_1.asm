dnl  x86-64 mpn_addmul_1 and mpn_submul_1, optimized for "Core 2".

dnl  Copyright 2003-2005, 2007-2009, 2011, 2012 Free Software Foundation, Inc.

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
C AMD K8,K9	 4
C AMD K10	 4
C AMD bd1	 5.1
C AMD bobcat
C Intel P4	 ?
C Intel core2	 4.3-4.5 (fluctuating)
C Intel NHM	 5.0
C Intel SBR	 4.1
C Intel atom	 ?
C VIA nano	 5.25

C INPUT PARAMETERS
define(`rp',	`%rdi')
define(`up',	`%rsi')
define(`n',	`%rdx')
define(`v0',	`%rcx')

ifdef(`OPERATION_addmul_1',`
      define(`ADDSUB',        `add')
      define(`func',     `mpn_addmul_1')
      define(`func_1c',  `mpn_addmul_1c')
')
ifdef(`OPERATION_submul_1',`
      define(`ADDSUB',        `sub')
      define(`func',     `mpn_submul_1')
      define(`func_1c',  `mpn_submul_1c')
')

MULFUNC_PROLOGUE(mpn_addmul_1 mpn_addmul_1c mpn_submul_1 mpn_submul_1c)

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

	C For DOS, on the stack we have four saved registers, return address,
	C space for four register arguments, and finally the carry input.

IFDOS(` define(`carry_in', `72(%rsp)')') dnl
IFSTD(` define(`carry_in', `%r8')') dnl

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(func_1c)
	FUNC_ENTRY(4)
	push	%rbx
	push	%rbp
	lea	(%rdx), %rbx
	neg	%rbx

	mov	(up), %rax
	mov	(rp), %r10

	lea	-16(rp,%rdx,8), rp
	lea	(up,%rdx,8), up
	mul	%rcx
	add	carry_in, %rax
	adc	$0, %rdx
	jmp	L(start_nc)
EPILOGUE()

	ALIGN(16)
PROLOGUE(func)
	FUNC_ENTRY(4)
	push	%rbx
	push	%rbp
	lea	(%rdx), %rbx
	neg	%rbx

	mov	(up), %rax
	mov	(rp), %r10

	lea	-16(rp,%rdx,8), rp
	lea	(up,%rdx,8), up
	mul	%rcx

L(start_nc):
	bt	$0, R32(%rbx)
	jc	L(odd)

	lea	(%rax), %r11
	mov	8(up,%rbx,8), %rax
	lea	(%rdx), %rbp
	mul	%rcx
	add	$2, %rbx
	jns	L(n2)

	lea	(%rax), %r8
	mov	(up,%rbx,8), %rax
	lea	(%rdx), %r9
	jmp	L(mid)

L(odd):	add	$1, %rbx
	jns	L(n1)

	lea	(%rax), %r8
	mov	(up,%rbx,8), %rax
	lea	(%rdx), %r9
	mul	%rcx
	lea	(%rax), %r11
	mov	8(up,%rbx,8), %rax
	lea	(%rdx), %rbp
	jmp	L(e)

	ALIGN(16)
L(top):	mul	%rcx
	ADDSUB	%r8, %r10
	lea	(%rax), %r8
	mov	(up,%rbx,8), %rax
	adc	%r9, %r11
	mov	%r10, -8(rp,%rbx,8)
	mov	(rp,%rbx,8), %r10
	lea	(%rdx), %r9
	adc	$0, %rbp
L(mid):	mul	%rcx
	ADDSUB	%r11, %r10
	lea	(%rax), %r11
	mov	8(up,%rbx,8), %rax
	adc	%rbp, %r8
	mov	%r10, (rp,%rbx,8)
	mov	8(rp,%rbx,8), %r10
	lea	(%rdx), %rbp
	adc	$0, %r9
L(e):	add	$2, %rbx
	js	L(top)

	mul	%rcx
	ADDSUB	%r8, %r10
	adc	%r9, %r11
	mov	%r10, -8(rp)
	adc	$0, %rbp
L(n2):	mov	(rp), %r10
	ADDSUB	%r11, %r10
	adc	%rbp, %rax
	mov	%r10, (rp)
	adc	$0, %rdx
L(n1):	mov	8(rp), %r10
	ADDSUB	%rax, %r10
	mov	%r10, 8(rp)
	mov	R32(%rbx), R32(%rax)	C zero rax
	adc	%rdx, %rax
	pop	%rbp
	pop	%rbx
	FUNC_EXIT()
	ret
EPILOGUE()
