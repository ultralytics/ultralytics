dnl  AMD64 mpn_bdiv_q_1, mpn_pi1_bdiv_q_1 -- schoolbook Hensel division by
dnl  1-limb divisor, returning quotient only.

dnl  Copyright 2001, 2002, 2004-2006, 2009, 2011, 2012 Free Software
dnl  Foundation, Inc.

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
C AMD K8,K9	10
C AMD K10	10
C Intel P4	33
C Intel core2	13.25
C Intel corei	14
C Intel atom	42
C VIA nano	 ?


C INPUT PARAMETERS
define(`rp',		`%rdi')
define(`up',		`%rsi')
define(`n',		`%rdx')
define(`d',		`%rcx')
define(`di',		`%r8')		C	just mpn_pi1_bdiv_q_1
define(`ncnt',		`%r9')		C	just mpn_pi1_bdiv_q_1

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_bdiv_q_1)
	FUNC_ENTRY(4)
	push	%rbx

	mov	%rcx, %rax
	xor	R32(%rcx), R32(%rcx)	C ncnt count
	mov	%rdx, %r10

	bt	$0, R32(%rax)
	jnc	L(evn)			C skip bsfq unless divisor is even

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

	lea	(%rax,%rax), %r8	C 2*inv
	imul	%rax, %rax		C inv*inv
	imul	%rbx, %rax		C inv*inv*d
	sub	%rax, %r8		C inv = 2*inv - inv*inv*d, 64 bits

	jmp	L(com)

L(evn):	bsf	%rax, %rcx
	shr	R8(%rcx), %rax
	jmp	L(odd)
EPILOGUE()

PROLOGUE(mpn_pi1_bdiv_q_1)
	FUNC_ENTRY(4)
IFDOS(`	mov	56(%rsp), %r8	')
IFDOS(`	mov	64(%rsp), %r9	')
	push	%rbx

	mov	%rcx, %r11		C d
	mov	%rdx, %r10		C n
	mov	%r9, %rcx		C ncnt

L(com):	mov	(up), %rax		C up[0]

	dec	%r10
	jz	L(one)

	mov	8(up), %rdx		C up[1]
	lea	(up,%r10,8), up		C up end
	lea	(rp,%r10,8), rp		C rp end
	neg	%r10			C -n

	shrd	R8(%rcx), %rdx, %rax

	xor	R32(%rbx), R32(%rbx)
	jmp	L(ent)

	ALIGN(8)
L(top):
	C rax	q
	C rbx	carry bit, 0 or 1
	C rcx	ncnt
	C rdx
	C r10	counter, limbs, negative

	mul	%r11			C carry limb in rdx
	mov	(up,%r10,8), %rax
	mov	8(up,%r10,8), %r9
	shrd	R8(%rcx), %r9, %rax
	nop
	sub	%rbx, %rax		C apply carry bit
	setc	R8(%rbx)
	sub	%rdx, %rax		C apply carry limb
	adc	$0, %rbx
L(ent):	imul	%r8, %rax
	mov	%rax, (rp,%r10,8)
	inc	%r10
	jnz	L(top)

	mul	%r11			C carry limb in rdx
	mov	(up), %rax		C up high limb
	shr	R8(%rcx), %rax
	sub	%rbx, %rax		C apply carry bit
	sub	%rdx, %rax		C apply carry limb
	imul	%r8, %rax
	mov	%rax, (rp)
	pop	%rbx
	FUNC_EXIT()
	ret

L(one):	shr	R8(%rcx), %rax
	imul	%r8, %rax
	mov	%rax, (rp)
	pop	%rbx
	FUNC_EXIT()
	ret
EPILOGUE()
