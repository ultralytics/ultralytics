dnl  AMD64 mpn_addmul_1 and mpn_submul_1 optimised for AMD bobcat.

dnl  Copyright 2003-2005, 2007, 2008, 2011, 2012 Free Software Foundation, Inc.

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
C AMD K10	 4.5
C AMD bd1	 4.75
C AMD bobcat	 5
C Intel P4	17.7
C Intel core2	 5.5
C Intel NHM	 5.43
C Intel SBR	 3.92
C Intel atom	23
C VIA nano	 5.63

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ifdef(`OPERATION_addmul_1',`
      define(`ADDSUB',        `add')
      define(`func',  `mpn_addmul_1')
')
ifdef(`OPERATION_submul_1',`
      define(`ADDSUB',        `sub')
      define(`func',  `mpn_submul_1')
')

MULFUNC_PROLOGUE(mpn_addmul_1 mpn_submul_1)

C Standard parameters
define(`rp',              `%rdi')
define(`up',              `%rsi')
define(`n_param',         `%rdx')
define(`v0',              `%rcx')
C Standard allocations
define(`n',               `%rbx')
define(`w0',              `%r8')
define(`w1',              `%r9')
define(`w2',              `%r10')
define(`w3',              `%r11')

C DOS64 parameters
IFDOS(` define(`rp',      `%rcx')    ') dnl
IFDOS(` define(`up',      `%rsi')    ') dnl
IFDOS(` define(`n_param', `%r8')     ') dnl
IFDOS(` define(`v0',      `%r9')     ') dnl
C DOS64 allocations
IFDOS(` define(`n',       `%rbx')    ') dnl
IFDOS(` define(`w0',      `%r8')     ') dnl
IFDOS(` define(`w1',      `%rdi')    ') dnl
IFDOS(` define(`w2',      `%r10')    ') dnl
IFDOS(` define(`w3',      `%r11')    ') dnl

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(func)
IFDOS(`	push	%rsi		')
IFDOS(`	push	%rdi		')
IFDOS(`	mov	%rdx, %rsi	')

	push	%rbx
	mov	(up), %rax

	lea	-16(rp,n_param,8), rp
	lea	-16(up,n_param,8), up

	mov	n_param, n
	and	$3, R32(n_param)
	jz	L(b0)
	cmp	$2, R32(n_param)
	ja	L(b3)
	jz	L(b2)

L(b1):	mul	v0
	cmp	$1, n
	jz	L(n1)
	mov	%rax, w2
	mov	%rdx, w3
	neg	n
	add	$3, n
	jmp	L(L1)
L(n1):	ADDSUB	%rax, 8(rp)
	adc	$0, %rdx
	mov	%rdx, %rax
	pop	%rbx
IFDOS(`	pop	%rdi		')
IFDOS(`	pop	%rsi		')
	ret

L(b3):	mul	v0
	mov	%rax, w2
	mov	%rdx, w3
	neg	n
	inc	n
	jmp	L(L3)

L(b0):	mul	v0
	mov	%rax, w0
	mov	%rdx, w1
	neg	n
	add	$2, n
	jmp	L(L0)

L(b2):	mul	v0
	mov	%rax, w0
	mov	%rdx, w1
	neg	n
	jmp	L(L2)

	ALIGN(16)
L(top):	ADDSUB	w0, -16(rp,n,8)
	adc	w1, w2
	adc	$0, w3
L(L1):	mov	0(up,n,8), %rax
	mul	v0
	mov	%rax, w0
	mov	%rdx, w1
	ADDSUB	w2, -8(rp,n,8)
	adc	w3, w0
	adc	$0, w1
L(L0):	mov	8(up,n,8), %rax
	mul	v0
	mov	%rax, w2
	mov	%rdx, w3
	ADDSUB	w0, 0(rp,n,8)
	adc	w1, w2
	adc	$0, w3
L(L3):	mov	16(up,n,8), %rax
	mul	v0
	mov	%rax, w0
	mov	%rdx, w1
	ADDSUB	w2, 8(rp,n,8)
	adc	w3, w0
	adc	$0, w1
L(L2):	mov	24(up,n,8), %rax
	mul	v0
	mov	%rax, w2
	mov	%rdx, w3
	add	$4, n
	js	L(top)

L(end):	ADDSUB	w0, (rp)
	adc	w1, w2
	adc	$0, w3
	ADDSUB	w2, 8(rp)
	adc	$0, w3
	mov	w3, %rax

	pop	%rbx
IFDOS(`	pop	%rdi		')
IFDOS(`	pop	%rsi		')
	ret
EPILOGUE()
