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
C AMD K8,K9	 ?
C AMD K10	 ?
C Intel P4	 ?
C Intel core2	 3.25
C Intel NHM	 4
C Intel SBR	 2  C (or 1.95 when L(top)'s alignment = 16 (mod 32))
C Intel atom	 ?
C VIA nano	 ?

C This code probably runs close to optimally on Sandy Bridge (using 4-way
C unrolling).  It also runs reasonably well on Core 2, but it runs poorly on
C all other processors, including Nehalem.

C INPUT PARAMETERS
define(`rp',	`%rdi')
define(`up',	`%rsi')
define(`vp',	`%rdx')
define(`n',	`%rcx')
define(`cy',	`%r8')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(func_nc)
	FUNC_ENTRY(4)
IFDOS(`	mov	56(%rsp), %r8	')
	push	%rbp
	mov	cy, %rax
	neg	%rax			C set msb on carry
	xor	R32(%rbp), R32(%rbp)	C limb carry
	mov	(vp), %r8
	shrd	$RSH, %r8, %rbp
	mov	R32(n), R32(%r9)
	and	$3, R32(%r9)
	je	L(b00)
	cmp	$2, R32(%r9)
	jc	L(b01)
	je	L(b10)
	jmp	L(b11)
EPILOGUE()

	ALIGN(16)
PROLOGUE(func_n)
	FUNC_ENTRY(4)
	push	%rbp
	xor	R32(%rbp), R32(%rbp)	C limb carry
	mov	(vp), %r8
	shrd	$RSH, %r8, %rbp
	mov	R32(n), R32(%rax)
	and	$3, R32(%rax)
	je	L(b00)
	cmp	$2, R32(%rax)
	jc	L(b01)
	je	L(b10)

L(b11):	mov	8(vp), %r9
	shrd	$RSH, %r9, %r8
	mov	16(vp), %r10
	shrd	$RSH, %r10, %r9
	add	R32(%rax), R32(%rax)	C init carry flag
	ADCSBB	(up), %rbp
	ADCSBB	8(up), %r8
	ADCSBB	16(up), %r9
	mov	%rbp, (rp)
	mov	%r8, 8(rp)
	mov	%r9, 16(rp)
	mov	%r10, %rbp
	lea	24(up), up
	lea	24(vp), vp
	lea	24(rp), rp
	sbb	R32(%rax), R32(%rax)	C save carry flag
	sub	$3, n
	ja	L(top)
	jmp	L(end)

L(b01):	add	R32(%rax), R32(%rax)	C init carry flag
	ADCSBB	(up), %rbp
	mov	%rbp, (rp)
	mov	%r8, %rbp
	lea	8(up), up
	lea	8(vp), vp
	lea	8(rp), rp
	sbb	R32(%rax), R32(%rax)	C save carry flag
	sub	$1, n
	ja	L(top)
	jmp	L(end)

L(b10):	mov	8(vp), %r9
	shrd	$RSH, %r9, %r8
	add	R32(%rax), R32(%rax)	C init carry flag
	ADCSBB	(up), %rbp
	ADCSBB	8(up), %r8
	mov	%rbp, (rp)
	mov	%r8, 8(rp)
	mov	%r9, %rbp
	lea	16(up), up
	lea	16(vp), vp
	lea	16(rp), rp
	sbb	R32(%rax), R32(%rax)	C save carry flag
	sub	$2, n
	ja	L(top)
	jmp	L(end)

	ALIGN(16)
L(top):	mov	(vp), %r8
	shrd	$RSH, %r8, %rbp
L(b00):	mov	8(vp), %r9
	shrd	$RSH, %r9, %r8
	mov	16(vp), %r10
	shrd	$RSH, %r10, %r9
	mov	24(vp), %r11
	shrd	$RSH, %r11, %r10
	lea	32(vp), vp
	add	R32(%rax), R32(%rax)	C restore carry flag
	ADCSBB	(up), %rbp
	ADCSBB	8(up), %r8
	ADCSBB	16(up), %r9
	ADCSBB	24(up), %r10
	lea	32(up), up
	mov	%rbp, (rp)
	mov	%r8, 8(rp)
	mov	%r9, 16(rp)
	mov	%r10, 24(rp)
	mov	%r11, %rbp
	lea	32(rp), rp
	sbb	R32(%rax), R32(%rax)	C save carry flag
	sub	$4, n
	jnz	L(top)

L(end):	shr	$RSH, %rbp
	add	R32(%rax), R32(%rax)	C restore carry flag
	ADCSBB	$0, %rbp
	mov	%rbp, %rax
	pop	%rbp
	FUNC_EXIT()
	ret
EPILOGUE()
