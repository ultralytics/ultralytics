dnl  AMD64 mpn_divexact_1 -- mpn by limb exact division.

dnl  Copyright 2001, 2002, 2004-2006, 2010-2012 Free Software Foundation, Inc.

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
C	       norm	       unorm
C AMD K8,K9	11		11
C AMD K10	11		11
C Intel P4	 ?
C Intel core2	13.5		13.25
C Intel corei	14.25
C Intel atom	34		36
C VIA nano	19.25		19.25


C INPUT PARAMETERS
C rp		rdi
C up		rsi
C n		rdx
C divisor	rcx

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_divexact_1)
	FUNC_ENTRY(4)
	push	%rbx

	mov	%rcx, %rax
	xor	R32(%rcx), R32(%rcx)	C shift count
	mov	%rdx, %r8

	bt	$0, R32(%rax)
	jc	L(odd)			C skip bsfq unless divisor is even
	bsf	%rax, %rcx
	shr	R8(%rcx), %rax
L(odd):	mov	%rax, %rbx
	shr	R32(%rax)
	and	$127, R32(%rax)		C d/2, 7 bits

	LEA(	binvert_limb_table, %rdx)

	movzbl	(%rdx,%rax), R32(%rax)	C inv 8 bits

	mov	%rbx, %r11		C d without twos

	lea	(%rax,%rax), R32(%rdx)	C 2*inv
	imul	R32(%rax), R32(%rax)	C inv*inv
	imul	R32(%rbx), R32(%rax)	C inv*inv*d
	sub	R32(%rax), R32(%rdx)	C inv = 2*inv - inv*inv*d, 16 bits

	lea	(%rdx,%rdx), R32(%rax)	C 2*inv
	imul	R32(%rdx), R32(%rdx)	C inv*inv
	imul	R32(%rbx), R32(%rdx)	C inv*inv*d
	sub	R32(%rdx), R32(%rax)	C inv = 2*inv - inv*inv*d, 32 bits

	lea	(%rax,%rax), %r10	C 2*inv
	imul	%rax, %rax		C inv*inv
	imul	%rbx, %rax		C inv*inv*d
	sub	%rax, %r10		C inv = 2*inv - inv*inv*d, 64 bits

	lea	(%rsi,%r8,8), %rsi	C up end
	lea	-8(%rdi,%r8,8), %rdi	C rp end
	neg	%r8			C -n

	mov	(%rsi,%r8,8), %rax	C up[0]

	inc	%r8
	jz	L(one)

	test	R32(%rcx), R32(%rcx)
	jnz	L(unorm)		C branch if count != 0
	xor	R32(%rbx), R32(%rbx)
	jmp	L(nent)

	ALIGN(8)
L(ntop):mul	%r11			C carry limb in rdx	0 10
	mov	-8(%rsi,%r8,8), %rax	C
	sub	%rbx, %rax		C apply carry bit
	setc	%bl			C
	sub	%rdx, %rax		C apply carry limb	5
	adc	$0, %rbx		C			6
L(nent):imul	%r10, %rax		C			6
	mov	%rax, (%rdi,%r8,8)	C
	inc	%r8			C
	jnz	L(ntop)

	mov	-8(%rsi), %r9		C up high limb
	jmp	L(com)

L(unorm):
	mov	(%rsi,%r8,8), %r9	C up[1]
	shr	R8(%rcx), %rax		C
	neg	R32(%rcx)
	shl	R8(%rcx), %r9		C
	neg	R32(%rcx)
	or	%r9, %rax
	xor	R32(%rbx), R32(%rbx)
	jmp	L(uent)

	ALIGN(8)
L(utop):mul	%r11			C carry limb in rdx	0 10
	mov	(%rsi,%r8,8), %rax	C
	shl	R8(%rcx), %rax		C
	neg	R32(%rcx)
	or	%r9, %rax
	sub	%rbx, %rax		C apply carry bit
	setc	%bl			C
	sub	%rdx, %rax		C apply carry limb	5
	adc	$0, %rbx		C			6
L(uent):imul	%r10, %rax		C			6
	mov	(%rsi,%r8,8), %r9	C
	shr	R8(%rcx), %r9		C
	neg	R32(%rcx)
	mov	%rax, (%rdi,%r8,8)	C
	inc	%r8			C
	jnz	L(utop)

L(com):	mul	%r11			C carry limb in rdx
	sub	%rbx, %r9		C apply carry bit
	sub	%rdx, %r9		C apply carry limb
	imul	%r10, %r9
	mov	%r9, (%rdi)
	pop	%rbx
	FUNC_EXIT()
	ret

L(one):	shr	R8(%rcx), %rax
	imul	%r10, %rax
	mov	%rax, (%rdi)
	pop	%rbx
	FUNC_EXIT()
	ret
EPILOGUE()
