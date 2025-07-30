dnl  AMD64 SSE mpn_sec_tabselect.

dnl  Contributed to the GNU project by Torbjörn Granlund.

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


C	     cycles/limb     cycles/limb     cycles/limb
C	      ali,evn n	     unal,evn n	      other cases
C AMD K8,K9	 1.65		1.65		 1.8
C AMD K10	 0.78		0.78		 0.85
C AMD bd1	 0.80		0.91		 1.25
C AMD bobcat	 2.15		2.15		 2.37
C Intel P4	 2.5		2.5		 2.95
C Intel core2	 1.17		1.25		 1.25
C Intel NHM	 0.87		0.90		 0.90
C Intel SBR	 0.63		0.79		 0.77
C Intel atom	 4.3		 4.3		 4.3	slower than plain code
C VIA nano	 1.4		 5.1		 3.14	too alignment dependent

C NOTES
C  * We only honour the least significant 32 bits of the `which' and `nents'
C    arguments to allow efficient code using just SSE2.  We would need to
C    either use the SSE4_1 pcmpeqq, or find some other SSE2 sequence.
C  * We use movd for copying between xmm and plain registers, since old gas
C    rejects movq.  But gas assembles movd as movq when given a 64-bit greg.

define(`rp',     `%rdi')
define(`tp',     `%rsi')
define(`n',      `%rdx')
define(`nents',  `%rcx')
define(`which',  `%r8')

define(`i',      `%r10')
define(`j',      `%r9')

C rax  rbx  rcx  rdx  rdi  rsi  rbp   r8   r9  r10  r11  r12  r13  r14  r15
C          nents  n   rp   tab       which j    i   temp  *    *    *    *

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_sec_tabselect)
	FUNC_ENTRY(4)
IFDOS(`	mov	56(%rsp), %r8d	')

	movd	which, %xmm8
	pshufd	$0, %xmm8, %xmm8	C 4 `which' copies
	mov	$1, R32(%rax)
	movd	%rax, %xmm9
	pshufd	$0, %xmm9, %xmm9	C 4 copies of 1

	mov	n, j
	add	$-8, j
	js	L(outer_end)

L(outer_top):
	mov	nents, i
	mov	tp, %r11
	pxor	%xmm13, %xmm13
	pxor	%xmm4, %xmm4
	pxor	%xmm5, %xmm5
	pxor	%xmm6, %xmm6
	pxor	%xmm7, %xmm7
	ALIGN(16)
L(top):	movdqa	%xmm8, %xmm0
	pcmpeqd	%xmm13, %xmm0
	paddd	%xmm9, %xmm13
	movdqu	0(tp), %xmm2
	movdqu	16(tp), %xmm3
	pand	%xmm0, %xmm2
	pand	%xmm0, %xmm3
	por	%xmm2, %xmm4
	por	%xmm3, %xmm5
	movdqu	32(tp), %xmm2
	movdqu	48(tp), %xmm3
	pand	%xmm0, %xmm2
	pand	%xmm0, %xmm3
	por	%xmm2, %xmm6
	por	%xmm3, %xmm7
	lea	(tp,n,8), tp
	add	$-1, i
	jne	L(top)

	movdqu	%xmm4, 0(rp)
	movdqu	%xmm5, 16(rp)
	movdqu	%xmm6, 32(rp)
	movdqu	%xmm7, 48(rp)

	lea	64(%r11), tp
	lea	64(rp), rp
	add	$-8, j
	jns	L(outer_top)
L(outer_end):

	test	$4, R8(n)
	je	L(b0xx)
L(b1xx):mov	nents, i
	mov	tp, %r11
	pxor	%xmm13, %xmm13
	pxor	%xmm4, %xmm4
	pxor	%xmm5, %xmm5
	ALIGN(16)
L(tp4):	movdqa	%xmm8, %xmm0
	pcmpeqd	%xmm13, %xmm0
	paddd	%xmm9, %xmm13
	movdqu	0(tp), %xmm2
	movdqu	16(tp), %xmm3
	pand	%xmm0, %xmm2
	pand	%xmm0, %xmm3
	por	%xmm2, %xmm4
	por	%xmm3, %xmm5
	lea	(tp,n,8), tp
	add	$-1, i
	jne	L(tp4)
	movdqu	%xmm4, 0(rp)
	movdqu	%xmm5, 16(rp)
	lea	32(%r11), tp
	lea	32(rp), rp

L(b0xx):test	$2, R8(n)
	je	L(b00x)
L(b01x):mov	nents, i
	mov	tp, %r11
	pxor	%xmm13, %xmm13
	pxor	%xmm4, %xmm4
	ALIGN(16)
L(tp2):	movdqa	%xmm8, %xmm0
	pcmpeqd	%xmm13, %xmm0
	paddd	%xmm9, %xmm13
	movdqu	0(tp), %xmm2
	pand	%xmm0, %xmm2
	por	%xmm2, %xmm4
	lea	(tp,n,8), tp
	add	$-1, i
	jne	L(tp2)
	movdqu	%xmm4, 0(rp)
	lea	16(%r11), tp
	lea	16(rp), rp

L(b00x):test	$1, R8(n)
	je	L(b000)
L(b001):mov	nents, i
	mov	tp, %r11
	pxor	%xmm13, %xmm13
	pxor	%xmm4, %xmm4
	ALIGN(16)
L(tp1):	movdqa	%xmm8, %xmm0
	pcmpeqd	%xmm13, %xmm0
	paddd	%xmm9, %xmm13
	movq	0(tp), %xmm2
	pand	%xmm0, %xmm2
	por	%xmm2, %xmm4
	lea	(tp,n,8), tp
	add	$-1, i
	jne	L(tp1)
	movq	%xmm4, 0(rp)

L(b000):FUNC_EXIT()
	ret
EPILOGUE()
