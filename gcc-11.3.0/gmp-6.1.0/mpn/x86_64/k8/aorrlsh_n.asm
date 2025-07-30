dnl  AMD64 mpn_addlsh_n and mpn_rsblsh_n.  R = V2^k +- U.

dnl  Copyright 2006, 2010-2012 Free Software Foundation, Inc.

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
C AMD K8,K9	 2.87	< 3.85 for lshift + add_n
C AMD K10	 2.75	< 3.85 for lshift + add_n
C Intel P4	22	> 7.33 for lshift + add_n
C Intel core2	 4.1	> 3.27 for lshift + add_n
C Intel NHM	 4.4	> 3.75 for lshift + add_n
C Intel SBR	 3.17	< 3.46 for lshift + add_n
C Intel atom	 ?	? 8.75 for lshift + add_n
C VIA nano	 4.7	< 6.25 for lshift + add_n

C TODO
C  * Can we propagate carry into rdx instead of using a special carry register?
C    That could save enough insns to get to 10 cycles/iteration.

define(`rp',       `%rdi')
define(`up',       `%rsi')
define(`vp_param', `%rdx')
define(`n_param',  `%rcx')
define(`cnt',      `%r8')

define(`vp',    `%r12')
define(`n',     `%rbp')

ifdef(`OPERATION_addlsh_n',`
  define(ADDSUB,       `add')
  define(ADCSBB,       `adc')
  define(func, mpn_addlsh_n)
')
ifdef(`OPERATION_rsblsh_n',`
  define(ADDSUB,       `sub')
  define(ADCSBB,       `sbb')
  define(func, mpn_rsblsh_n)
')

MULFUNC_PROLOGUE(mpn_addlsh_n mpn_rsblsh_n)

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(func)
	FUNC_ENTRY(4)
IFDOS(`	mov	56(%rsp), %r8d	')
	push	%r12
	push	%rbp
	push	%rbx

	mov	(vp_param), %rax	C load first V limb early

	mov	$0, R32(n)
	sub	n_param, n

	lea	-16(up,n_param,8), up
	lea	-16(rp,n_param,8), rp
	lea	16(vp_param,n_param,8), vp

	mov	n_param, %r9

	mov	%r8, %rcx
	mov	$1, R32(%r8)
	shl	R8(%rcx), %r8

	mul	%r8			C initial multiply

	and	$3, R32(%r9)
	jz	L(b0)
	cmp	$2, R32(%r9)
	jc	L(b1)
	jz	L(b2)

L(b3):	mov	%rax, %r11
	ADDSUB	16(up,n,8), %r11
	mov	-8(vp,n,8), %rax
	sbb	R32(%rcx), R32(%rcx)
	mov	%rdx, %rbx
	mul	%r8
	or	%rax, %rbx
	mov	(vp,n,8), %rax
	mov	%rdx, %r9
	mul	%r8
	or	%rax, %r9
	add	$3, n
	jnz	L(lo3)
	jmp	L(cj3)

L(b2):	mov	%rax, %rbx
	mov	-8(vp,n,8), %rax
	mov	%rdx, %r9
	mul	%r8
	or	%rax, %r9
	add	$2, n
	jz	L(cj2)
	mov	%rdx, %r10
	mov	-16(vp,n,8), %rax
	mul	%r8
	or	%rax, %r10
	xor	R32(%rcx), R32(%rcx)	C clear carry register
	jmp	L(lo2)

L(b1):	mov	%rax, %r9
	mov	%rdx, %r10
	add	$1, n
	jnz	L(gt1)
	ADDSUB	8(up,n,8), %r9
	jmp	L(cj1)
L(gt1):	mov	-16(vp,n,8), %rax
	mul	%r8
	or	%rax, %r10
	mov	%rdx, %r11
	mov	-8(vp,n,8), %rax
	mul	%r8
	or	%rax, %r11
	ADDSUB	8(up,n,8), %r9
	ADCSBB	16(up,n,8), %r10
	ADCSBB	24(up,n,8), %r11
	mov	(vp,n,8), %rax
	sbb	R32(%rcx), R32(%rcx)
	jmp	L(lo1)

L(b0):	mov	%rax, %r10
	mov	%rdx, %r11
	mov	-8(vp,n,8), %rax
	mul	%r8
	or	%rax, %r11
	ADDSUB	16(up,n,8), %r10
	ADCSBB	24(up,n,8), %r11
	mov	(vp,n,8), %rax
	sbb	R32(%rcx), R32(%rcx)
	mov	%rdx, %rbx
	mul	%r8
	or	%rax, %rbx
	mov	8(vp,n,8), %rax
	add	$4, n
	jz	L(end)

	ALIGN(8)
L(top):	mov	%rdx, %r9
	mul	%r8
	or	%rax, %r9
	mov	%r10, -16(rp,n,8)
L(lo3):	mov	%rdx, %r10
	mov	-16(vp,n,8), %rax
	mul	%r8
	or	%rax, %r10
	mov	%r11, -8(rp,n,8)
L(lo2):	mov	%rdx, %r11
	mov	-8(vp,n,8), %rax
	mul	%r8
	or	%rax, %r11
	add	R32(%rcx), R32(%rcx)
	ADCSBB	(up,n,8), %rbx
	ADCSBB	8(up,n,8), %r9
	ADCSBB	16(up,n,8), %r10
	ADCSBB	24(up,n,8), %r11
	mov	(vp,n,8), %rax
	sbb	R32(%rcx), R32(%rcx)
	mov	%rbx, (rp,n,8)
L(lo1):	mov	%rdx, %rbx
	mul	%r8
	or	%rax, %rbx
	mov	%r9, 8(rp,n,8)
L(lo0):	mov	8(vp,n,8), %rax
	add	$4, n
	jnz	L(top)

L(end):	mov	%rdx, %r9
	mul	%r8
	or	%rax, %r9
	mov	%r10, -16(rp,n,8)
L(cj3):	mov	%r11, -8(rp,n,8)
L(cj2):	add	R32(%rcx), R32(%rcx)
	ADCSBB	(up,n,8), %rbx
	ADCSBB	8(up,n,8), %r9
	mov	%rbx, (rp,n,8)
L(cj1):	mov	%r9, 8(rp,n,8)
	mov	%rdx, %rax
	ADCSBB	$0, %rax
	pop	%rbx
	pop	%rbp
	pop	%r12
	FUNC_EXIT()
	ret
EPILOGUE()
