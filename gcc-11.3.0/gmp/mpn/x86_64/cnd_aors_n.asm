dnl  AMD64 mpn_cnd_add_n, mpn_cnd_sub_n

dnl  Copyright 2011-2013 Free Software Foundation, Inc.

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
C AMD K8,K9	 2
C AMD K10	 2
C AMD bd1	 2.32
C AMD bobcat	 3
C Intel P4	13
C Intel core2	 2.9
C Intel NHM	 2.8
C Intel SBR	 2.4
C Intel atom	 5.33
C VIA nano	 3

C NOTES
C  * It might seem natural to use the cmov insn here, but since this function
C    is supposed to have the exact same execution pattern for cnd true and
C    false, and since cmov's documentation is not clear about whether it
C    actually reads both source operands and writes the register for a false
C    condition, we cannot use it.
C  * Two cases could be optimised: (1) cnd_add_n could use ADCSBB-from-memory
C    to save one insn/limb, and (2) when up=rp cnd_add_n and cnd_sub_n could use
C    ADCSBB-to-memory, again saving 1 insn/limb.
C  * This runs optimally at decoder bandwidth on K10.  It has not been tuned
C    for any other processor.

C INPUT PARAMETERS
define(`cnd',	`%rdi')	dnl rcx
define(`rp',	`%rsi')	dnl rdx
define(`up',	`%rdx')	dnl r8
define(`vp',	`%rcx')	dnl r9
define(`n',	`%r8')	dnl rsp+40

ifdef(`OPERATION_cnd_add_n', `
	define(ADDSUB,	      add)
	define(ADCSBB,	      adc)
	define(func,	      mpn_cnd_add_n)')
ifdef(`OPERATION_cnd_sub_n', `
	define(ADDSUB,	      sub)
	define(ADCSBB,	      sbb)
	define(func,	      mpn_cnd_sub_n)')

MULFUNC_PROLOGUE(mpn_cnd_add_n mpn_cnd_sub_n)

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(func)
	FUNC_ENTRY(4)
IFDOS(`	mov	56(%rsp), R32(%r8)')
	push	%rbx
	push	%rbp
	push	%r12
	push	%r13
	push	%r14

	neg	cnd
	sbb	cnd, cnd		C make cnd mask

	lea	(vp,n,8), vp
	lea	(up,n,8), up
	lea	(rp,n,8), rp

	mov	R32(n), R32(%rax)
	neg	n
	and	$3, R32(%rax)
	jz	L(top)			C carry-save reg rax = 0 in this arc
	cmp	$2, R32(%rax)
	jc	L(b1)
	jz	L(b2)

L(b3):	mov	(vp,n,8), %r12
	mov	8(vp,n,8), %r13
	mov	16(vp,n,8), %r14
	and	cnd, %r12
	mov	(up,n,8), %r10
	and	cnd, %r13
	mov	8(up,n,8), %rbx
	and	cnd, %r14
	mov	16(up,n,8), %rbp
	ADDSUB	%r12, %r10
	mov	%r10, (rp,n,8)
	ADCSBB	%r13, %rbx
	mov	%rbx, 8(rp,n,8)
	ADCSBB	%r14, %rbp
	mov	%rbp, 16(rp,n,8)
	sbb	R32(%rax), R32(%rax)	C save carry
	add	$3, n
	js	L(top)
	jmp	L(end)

L(b2):	mov	(vp,n,8), %r12
	mov	8(vp,n,8), %r13
	mov	(up,n,8), %r10
	and	cnd, %r12
	mov	8(up,n,8), %rbx
	and	cnd, %r13
	ADDSUB	%r12, %r10
	mov	%r10, (rp,n,8)
	ADCSBB	%r13, %rbx
	mov	%rbx, 8(rp,n,8)
	sbb	R32(%rax), R32(%rax)	C save carry
	add	$2, n
	js	L(top)
	jmp	L(end)

L(b1):	mov	(vp,n,8), %r12
	mov	(up,n,8), %r10
	and	cnd, %r12
	ADDSUB	%r12, %r10
	mov	%r10, (rp,n,8)
	sbb	R32(%rax), R32(%rax)	C save carry
	add	$1, n
	jns	L(end)

	ALIGN(16)
L(top):	mov	(vp,n,8), %r12
	mov	8(vp,n,8), %r13
	mov	16(vp,n,8), %r14
	mov	24(vp,n,8), %r11
	and	cnd, %r12
	mov	(up,n,8), %r10
	and	cnd, %r13
	mov	8(up,n,8), %rbx
	and	cnd, %r14
	mov	16(up,n,8), %rbp
	and	cnd, %r11
	mov	24(up,n,8), %r9
	add	R32(%rax), R32(%rax)	C restore carry
	ADCSBB	%r12, %r10
	mov	%r10, (rp,n,8)
	ADCSBB	%r13, %rbx
	mov	%rbx, 8(rp,n,8)
	ADCSBB	%r14, %rbp
	mov	%rbp, 16(rp,n,8)
	ADCSBB	%r11, %r9
	mov	%r9, 24(rp,n,8)
	sbb	R32(%rax), R32(%rax)	C save carry
	add	$4, n
	js	L(top)

L(end):	neg	R32(%rax)
	pop	%r14
	pop	%r13
	pop	%r12
	pop	%rbp
	pop	%rbx
	FUNC_EXIT()
	ret
EPILOGUE()
