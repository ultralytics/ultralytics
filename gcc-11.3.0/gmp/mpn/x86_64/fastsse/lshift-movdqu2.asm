dnl  AMD64 mpn_lshift optimised for CPUs with fast SSE including fast movdqu.

dnl  Contributed to the GNU project by Torbjorn Granlund.

dnl  Copyright 2010-2012 Free Software Foundation, Inc.

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


C	     cycles/limb     cycles/limb     cycles/limb    good
C              aligned	      unaligned	      best seen	   for cpu?
C AMD K8,K9	 3		 3		 2.35	  no, use shl/shr
C AMD K10	 1.5-1.8	 1.5-1.8	 1.33	  yes
C AMD bd1	 1.7-1.9	 1.7-1.9	 1.33	  yes
C AMD bobcat	 3.17		 3.17			  yes, bad for n < 20
C Intel P4	 4.67		 4.67		 2.7	  no, slow movdqu
C Intel core2	 2.15		 2.15		 1.25	  no, use shld/shrd
C Intel NHM	 1.66		 1.66		 1.25	  no, use shld/shrd
C Intel SBR	 1.3		 1.3		 1.25	  yes, bad for n = 4-6
C Intel atom	11.7		11.7		 4.5	  no
C VIA nano	 5.7		 5.95		 2.0	  no, slow movdqu

C We try to do as many aligned 16-byte operations as possible.  The top-most
C and bottom-most writes might need 8-byte operations.
C
C This variant rely on fast load movdqu, and uses it even for aligned operands,
C in order to avoid the need for two separate loops.
C
C TODO
C  * Could 2-limb wind-down code be simplified?
C  * Improve basecase code, using shld/shrd for SBR, discrete integer shifts
C    for other affected CPUs.

C INPUT PARAMETERS
define(`rp',  `%rdi')
define(`ap',  `%rsi')
define(`n',   `%rdx')
define(`cnt', `%rcx')

ASM_START()
	TEXT
	ALIGN(64)
PROLOGUE(mpn_lshift)
	FUNC_ENTRY(4)
	movd	R32(%rcx), %xmm4
	mov	$64, R32(%rax)
	sub	R32(%rcx), R32(%rax)
	movd	R32(%rax), %xmm5

	neg	R32(%rcx)
	mov	-8(ap,n,8), %rax
	shr	R8(%rcx), %rax

	cmp	$3, n
	jle	L(bc)

	lea	(rp,n,8), R32(%rcx)
	test	$8, R8(%rcx)
	jz	L(rp_aligned)

C Do one initial limb in order to make rp aligned
	movq	-8(ap,n,8), %xmm0
	movq	-16(ap,n,8), %xmm1
	psllq	%xmm4, %xmm0
	psrlq	%xmm5, %xmm1
	por	%xmm1, %xmm0
	movq	%xmm0, -8(rp,n,8)
	dec	n

L(rp_aligned):
	lea	1(n), %r8d

	and	$6, R32(%r8)
	jz	L(ba0)
	cmp	$4, R32(%r8)
	jz	L(ba4)
	jc	L(ba2)
L(ba6):	add	$-4, n
	jmp	L(i56)
L(ba0):	add	$-6, n
	jmp	L(i70)
L(ba4):	add	$-2, n
	jmp	L(i34)
L(ba2):	add	$-8, n
	jle	L(end)

	ALIGN(16)
L(top):	movdqu	40(ap,n,8), %xmm1
	movdqu	48(ap,n,8), %xmm0
	psllq	%xmm4, %xmm0
	psrlq	%xmm5, %xmm1
	por	%xmm1, %xmm0
	movdqa	%xmm0, 48(rp,n,8)
L(i70):
	movdqu	24(ap,n,8), %xmm1
	movdqu	32(ap,n,8), %xmm0
	psllq	%xmm4, %xmm0
	psrlq	%xmm5, %xmm1
	por	%xmm1, %xmm0
	movdqa	%xmm0, 32(rp,n,8)
L(i56):
	movdqu	8(ap,n,8), %xmm1
	movdqu	16(ap,n,8), %xmm0
	psllq	%xmm4, %xmm0
	psrlq	%xmm5, %xmm1
	por	%xmm1, %xmm0
	movdqa	%xmm0, 16(rp,n,8)
L(i34):
	movdqu	-8(ap,n,8), %xmm1
	movdqu	(ap,n,8), %xmm0
	psllq	%xmm4, %xmm0
	psrlq	%xmm5, %xmm1
	por	%xmm1, %xmm0
	movdqa	%xmm0, (rp,n,8)
	sub	$8, n
	jg	L(top)

L(end):	test	$1, R8(n)
	jnz	L(end8)

	movdqu	(ap), %xmm1
	pxor	%xmm0, %xmm0
	punpcklqdq  %xmm1, %xmm0
	psllq	%xmm4, %xmm1
	psrlq	%xmm5, %xmm0
	por	%xmm1, %xmm0
	movdqa	%xmm0, (rp)
	FUNC_EXIT()
	ret

C Basecase
	ALIGN(16)
L(bc):	dec	R32(n)
	jz	L(end8)

	movq	(ap,n,8), %xmm1
	movq	-8(ap,n,8), %xmm0
	psllq	%xmm4, %xmm1
	psrlq	%xmm5, %xmm0
	por	%xmm1, %xmm0
	movq	%xmm0, (rp,n,8)
	sub	$2, R32(n)
	jl	L(end8)
	movq	8(ap), %xmm1
	movq	(ap), %xmm0
	psllq	%xmm4, %xmm1
	psrlq	%xmm5, %xmm0
	por	%xmm1, %xmm0
	movq	%xmm0, 8(rp)

L(end8):movq	(ap), %xmm0
	psllq	%xmm4, %xmm0
	movq	%xmm0, (rp)
	FUNC_EXIT()
	ret
EPILOGUE()
