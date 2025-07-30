dnl  AMD64 mpn_mod_1s_2p

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
C AMD K8,K9	 4
C AMD K10	 4
C Intel P4	19
C Intel core2	 8
C Intel NHM	 6.5
C Intel SBR	 4.5
C Intel atom	28
C VIA nano	 8

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_mod_1s_2p)
	FUNC_ENTRY(4)
	push	%r14
	test	$1, R8(%rsi)
	mov	%rdx, %r14
	push	%r13
	mov	%rcx, %r13
	push	%r12
	push	%rbp
	push	%rbx
	mov	16(%rcx), %r10
	mov	24(%rcx), %rbx
	mov	32(%rcx), %rbp
	je	L(b0)
	dec	%rsi
	je	L(one)
	mov	-8(%rdi,%rsi,8), %rax
	mul	%r10
	mov	%rax, %r9
	mov	%rdx, %r8
	mov	(%rdi,%rsi,8), %rax
	add	-16(%rdi,%rsi,8), %r9
	adc	$0, %r8
	mul	%rbx
	add	%rax, %r9
	adc	%rdx, %r8
	jmp	L(11)

L(b0):	mov	-8(%rdi,%rsi,8), %r8
	mov	-16(%rdi,%rsi,8), %r9

L(11):	sub	$4, %rsi
	jb	L(ed2)
	lea	40(%rdi,%rsi,8), %rdi
	mov	-40(%rdi), %r11
	mov	-32(%rdi), %rax
	jmp	L(m0)

	ALIGN(16)
L(top):	mov	-24(%rdi), %r9
	add	%rax, %r11
	mov	-16(%rdi), %rax
	adc	%rdx, %r12
	mul	%r10
	add	%rax, %r9
	mov	%r11, %rax
	mov	%rdx, %r8
	adc	$0, %r8
	mul	%rbx
	add	%rax, %r9
	mov	%r12, %rax
	adc	%rdx, %r8
	mul	%rbp
	sub	$2, %rsi
	jb	L(ed1)
	mov	-40(%rdi), %r11
	add	%rax, %r9
	mov	-32(%rdi), %rax
	adc	%rdx, %r8
L(m0):	mul	%r10
	add	%rax, %r11
	mov	%r9, %rax
	mov	%rdx, %r12
	adc	$0, %r12
	mul	%rbx
	add	%rax, %r11
	lea	-32(%rdi), %rdi		C ap -= 4
	mov	%r8, %rax
	adc	%rdx, %r12
	mul	%rbp
	sub	$2, %rsi
	jae	L(top)

L(ed0):	mov	%r11, %r9
	mov	%r12, %r8
L(ed1):	add	%rax, %r9
	adc	%rdx, %r8
L(ed2):	mov	8(%r13), R32(%rdi)		C cnt
	mov	%r8, %rax
	mov	%r9, %r8
	mul	%r10
	add	%rax, %r8
	adc	$0, %rdx
L(1):	xor	R32(%rcx), R32(%rcx)
	mov	%r8, %r9
	sub	R32(%rdi), R32(%rcx)
	shr	R8(%rcx), %r9
	mov	R32(%rdi), R32(%rcx)
	sal	R8(%rcx), %rdx
	or	%rdx, %r9
	sal	R8(%rcx), %r8
	mov	%r9, %rax
	mulq	(%r13)
	mov	%rax, %rsi
	inc	%r9
	add	%r8, %rsi
	adc	%r9, %rdx
	imul	%r14, %rdx
	sub	%rdx, %r8
	lea	(%r8,%r14), %rax
	cmp	%r8, %rsi
	cmovc	%rax, %r8
	mov	%r8, %rax
	sub	%r14, %rax
	cmovc	%r8, %rax
	mov	R32(%rdi), R32(%rcx)
	shr	R8(%rcx), %rax
	pop	%rbx
	pop	%rbp
	pop	%r12
	pop	%r13
	pop	%r14
	FUNC_EXIT()
	ret
L(one):
	mov	(%rdi), %r8
	mov	8(%rcx), R32(%rdi)
	xor	%rdx, %rdx
	jmp	L(1)
EPILOGUE()

	ALIGN(16)
PROLOGUE(mpn_mod_1s_2p_cps)
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
	add	%rdx, %r12
	cmp	%rdx, %rax
	cmovnc	%rdx, %r12

	shr	R8(%rcx), %r12
	mov	%r12, 32(%rbx)		C store B3modb

	pop	%r12
	pop	%rbx
	pop	%rbp
	FUNC_EXIT()
	ret
EPILOGUE()
