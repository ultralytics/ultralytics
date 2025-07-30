dnl  AMD64 mpn_mod_1s_4p

dnl  Contributed to the GNU project by Torbjorn Granlund.

dnl  Copyright 2009-2012, 2014 Free Software Foundation, Inc.

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
C AMD K8,K9	 3
C AMD K10	 3
C Intel P4	15.5
C Intel core2	 5
C Intel corei	 4
C Intel atom	23
C VIA nano	 4.75

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_mod_1s_4p)
	FUNC_ENTRY(4)
	push	%r15
	push	%r14
	push	%r13
	push	%r12
	push	%rbp
	push	%rbx

	mov	%rdx, %r15
	mov	%rcx, %r14
	mov	16(%rcx), %r11		C B1modb
	mov	24(%rcx), %rbx		C B2modb
	mov	32(%rcx), %rbp		C B3modb
	mov	40(%rcx), %r13		C B4modb
	mov	48(%rcx), %r12		C B5modb
	xor	R32(%r8), R32(%r8)
	mov	R32(%rsi), R32(%rdx)
	and	$3, R32(%rdx)
	je	L(b0)
	cmp	$2, R32(%rdx)
	jc	L(b1)
	je	L(b2)

L(b3):	lea	-24(%rdi,%rsi,8), %rdi
	mov	8(%rdi), %rax
	mul	%r11
	mov	(%rdi), %r9
	add	%rax, %r9
	adc	%rdx, %r8
	mov	16(%rdi), %rax
	mul	%rbx
	jmp	L(m0)

	ALIGN(8)
L(b0):	lea	-32(%rdi,%rsi,8), %rdi
	mov	8(%rdi), %rax
	mul	%r11
	mov	(%rdi), %r9
	add	%rax, %r9
	adc	%rdx, %r8
	mov	16(%rdi), %rax
	mul	%rbx
	add	%rax, %r9
	adc	%rdx, %r8
	mov	24(%rdi), %rax
	mul	%rbp
	jmp	L(m0)

	ALIGN(8)
L(b1):	lea	-8(%rdi,%rsi,8), %rdi
	mov	(%rdi), %r9
	jmp	L(m1)

	ALIGN(8)
L(b2):	lea	-16(%rdi,%rsi,8), %rdi
	mov	8(%rdi), %r8
	mov	(%rdi), %r9
	jmp	L(m1)

	ALIGN(16)
L(top):	mov	-24(%rdi), %rax
	mov	-32(%rdi), %r10
	mul	%r11			C up[1] * B1modb
	add	%rax, %r10
	mov	-16(%rdi), %rax
	mov	$0, R32(%rcx)
	adc	%rdx, %rcx
	mul	%rbx			C up[2] * B2modb
	add	%rax, %r10
	mov	-8(%rdi), %rax
	adc	%rdx, %rcx
	sub	$32, %rdi
	mul	%rbp			C up[3] * B3modb
	add	%rax, %r10
	mov	%r13, %rax
	adc	%rdx, %rcx
	mul	%r9			C rl * B4modb
	add	%rax, %r10
	mov	%r12, %rax
	adc	%rdx, %rcx
	mul	%r8			C rh * B5modb
	mov	%r10, %r9
	mov	%rcx, %r8
L(m0):	add	%rax, %r9
	adc	%rdx, %r8
L(m1):	sub	$4, %rsi
	ja	L(top)

L(end):	mov	8(%r14), R32(%rsi)
	mov	%r8, %rax
	mul	%r11
	mov	%rax, %r8
	add	%r9, %r8
	adc	$0, %rdx
	xor	R32(%rcx), R32(%rcx)
	sub	R32(%rsi), R32(%rcx)
	mov	%r8, %rdi
	shr	R8(%rcx), %rdi
	mov	R32(%rsi), R32(%rcx)
	sal	R8(%rcx), %rdx
	or	%rdx, %rdi
	mov	%rdi, %rax
	mulq	(%r14)
	mov	%r15, %rbx
	mov	%rax, %r9
	sal	R8(%rcx), %r8
	inc	%rdi
	add	%r8, %r9
	adc	%rdi, %rdx
	imul	%rbx, %rdx
	sub	%rdx, %r8
	lea	(%r8,%rbx), %rax
	cmp	%r8, %r9
	cmovc	%rax, %r8
	mov	%r8, %rax
	sub	%rbx, %rax
	cmovc	%r8, %rax
	shr	R8(%rcx), %rax
	pop	%rbx
	pop	%rbp
	pop	%r12
	pop	%r13
	pop	%r14
	pop	%r15
	FUNC_EXIT()
	ret
EPILOGUE()

	ALIGN(16)
PROLOGUE(mpn_mod_1s_4p_cps)
	FUNC_ENTRY(2)
	push	%rbp
	bsr	%rsi, %rcx
	push	%rbx
	mov	%rdi, %rbx
	push	%r12
	xor	$63, R32(%rcx)
	mov	%rsi, %r12
	mov	R32(%rcx), R32(%rbp)	C preserve cnt over call
	sal	R8(%rcx), %r12		C b << cnt
IFSTD(`	mov	%r12, %rdi	')	C pass parameter
IFDOS(`	mov	%r12, %rcx	')	C pass parameter
	ASSERT(nz, `test $15, %rsp')
	CALL(	mpn_invert_limb)
	mov	%r12, %r8
	mov	%rax, %r11
	mov	%rax, (%rbx)		C store bi
	mov	%rbp, 8(%rbx)		C store cnt
	neg	%r8
	mov	R32(%rbp), R32(%rcx)
	mov	$1, R32(%rsi)
ifdef(`SHLD_SLOW',`
	shl	R8(%rcx), %rsi
	neg	R32(%rcx)
	mov	%rax, %rbp
	shr	R8(%rcx), %rax
	or	%rax, %rsi
	mov	%rbp, %rax
	neg	R32(%rcx)
',`
	shld	R8(%rcx), %rax, %rsi	C FIXME: Slow on Atom and Nano
')
	imul	%r8, %rsi
	mul	%rsi

	add	%rsi, %rdx
	shr	R8(%rcx), %rsi
	mov	%rsi, 16(%rbx)		C store B1modb

	not	%rdx
	imul	%r12, %rdx
	lea	(%rdx,%r12), %rsi
	cmp	%rdx, %rax
	cmovnc	%rdx, %rsi
	mov	%r11, %rax
	mul	%rsi

	add	%rsi, %rdx
	shr	R8(%rcx), %rsi
	mov	%rsi, 24(%rbx)		C store B2modb

	not	%rdx
	imul	%r12, %rdx
	lea	(%rdx,%r12), %rsi
	cmp	%rdx, %rax
	cmovnc	%rdx, %rsi
	mov	%r11, %rax
	mul	%rsi

	add	%rsi, %rdx
	shr	R8(%rcx), %rsi
	mov	%rsi, 32(%rbx)		C store B3modb

	not	%rdx
	imul	%r12, %rdx
	lea	(%rdx,%r12), %rsi
	cmp	%rdx, %rax
	cmovnc	%rdx, %rsi
	mov	%r11, %rax
	mul	%rsi

	add	%rsi, %rdx
	shr	R8(%rcx), %rsi
	mov	%rsi, 40(%rbx)		C store B4modb

	not	%rdx
	imul	%r12, %rdx
	add	%rdx, %r12
	cmp	%rdx, %rax
	cmovnc	%rdx, %r12

	shr	R8(%rcx), %r12
	mov	%r12, 48(%rbx)		C store B5modb

	pop	%r12
	pop	%rbx
	pop	%rbp
	FUNC_EXIT()
	ret
EPILOGUE()
