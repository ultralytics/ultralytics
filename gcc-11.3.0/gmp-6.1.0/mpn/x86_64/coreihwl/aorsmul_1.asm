dnl  AMD64 mpn_addmul_1 and mpn_submul_1 optimised for Intel Haswell.

dnl  Contributed to the GNU project by Torbjörn Granlund.

dnl  Copyright 2013 Free Software Foundation, Inc.

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
C AMD K8,K9	n/a
C AMD K10	n/a
C AMD bull	n/a
C AMD pile	n/a
C AMD steam	 ?
C AMD bobcat	n/a
C AMD jaguar	 ?
C Intel P4	n/a
C Intel core	n/a
C Intel NHM	n/a
C Intel SBR	n/a
C Intel IBR	n/a
C Intel HWL	 2.32
C Intel BWL	 ?
C Intel atom	n/a
C VIA nano	n/a

C The loop of this code is the result of running a code generation and
C optimisation tool suite written by David Harvey and Torbjörn Granlund.

C TODO
C  * Handle small n separately, for lower overhead.

define(`rp',      `%rdi')   C rcx
define(`up',      `%rsi')   C rdx
define(`n_param', `%rdx')   C r8
define(`v0_param',`%rcx')   C r9

define(`n',       `%rbp')
define(`v0',      `%rdx')

ifdef(`OPERATION_addmul_1',`
  define(`ADDSUB',        `add')
  define(`ADCSBB',        `adc')
  define(`func',  `mpn_addmul_1')
')
ifdef(`OPERATION_submul_1',`
  define(`ADDSUB',        `sub')
  define(`ADCSBB',        `sbb')
  define(`func',  `mpn_submul_1')
')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

MULFUNC_PROLOGUE(mpn_addmul_1 mpn_submul_1)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(func)
	FUNC_ENTRY(4)
	push	%rbx
	push	%rbp
	push	%r12
	push	%r13

	mov	n_param, n
	mov	v0_param, v0

	test	$1, R8(n)
	jnz	L(bx1)

L(bx0):	shr	$2, n
	jc	L(b10)

L(b00):	mulx(	(up), %r13, %r12)
	mulx(	8,(up), %rbx, %rax)
	add	%r12, %rbx
	adc	$0, %rax
	mov	(rp), %r12
	mov	8(rp), %rcx
	mulx(	16,(up), %r9, %r8)
	lea	-16(rp), rp
	lea	16(up), up
	ADDSUB	%r13, %r12
	jmp	L(lo0)

L(bx1):	shr	$2, n
	jc	L(b11)

L(b01):	mulx(	(up), %r11, %r10)
	jnz	L(gt1)
L(n1):	ADDSUB	%r11, (rp)
	mov	$0, R32(%rax)
	adc	%r10, %rax
	jmp	L(ret)

L(gt1):	mulx(	8,(up), %r13, %r12)
	mulx(	16,(up), %rbx, %rax)
	lea	24(up), up
	add	%r10, %r13
	adc	%r12, %rbx
	adc	$0, %rax
	mov	(rp), %r10
	mov	8(rp), %r12
	mov	16(rp), %rcx
	lea	-8(rp), rp
	ADDSUB	%r11, %r10
	jmp	L(lo1)

L(b11):	mulx(	(up), %rbx, %rax)
	mov	(rp), %rcx
	mulx(	8,(up), %r9, %r8)
	lea	8(up), up
	lea	-24(rp), rp
	inc	n			C adjust n
	ADDSUB	%rbx, %rcx
	jmp	L(lo3)

L(b10):	mulx(	(up), %r9, %r8)
	mulx(	8,(up), %r11, %r10)
	lea	-32(rp), rp
	mov	$0, R32(%rax)
	clc				C clear cf
	jz	L(end)			C depends on old shift

	ALIGN(16)
L(top):	adc	%rax, %r9
	lea	32(rp), rp
	adc	%r8, %r11
	mulx(	16,(up), %r13, %r12)
	mov	(rp), %r8
	mulx(	24,(up), %rbx, %rax)
	lea	32(up), up
	adc	%r10, %r13
	adc	%r12, %rbx
	adc	$0, %rax
	mov	8(rp), %r10
	mov	16(rp), %r12
	ADDSUB	%r9, %r8
	mov	24(rp), %rcx
	mov	%r8, (rp)
	ADCSBB	%r11, %r10
L(lo1):	mulx(	(up), %r9, %r8)
	mov	%r10, 8(rp)
	ADCSBB	%r13, %r12
L(lo0):	mov	%r12, 16(rp)
	ADCSBB	%rbx, %rcx
L(lo3):	mulx(	8,(up), %r11, %r10)
	mov	%rcx, 24(rp)
	dec	n
	jnz	L(top)

L(end):	adc	%rax, %r9
	adc	%r8, %r11
	mov	32(rp), %r8
	mov	%r10, %rax
	adc	$0, %rax
	mov	40(rp), %r10
	ADDSUB	%r9, %r8
	mov	%r8, 32(rp)
	ADCSBB	%r11, %r10
	mov	%r10, 40(rp)
	adc	$0, %rax

L(ret):	pop	%r13
	pop	%r12
	pop	%rbp
	pop	%rbx
	FUNC_EXIT()
	ret
EPILOGUE()
