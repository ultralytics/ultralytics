dnl  AMD64 mpn_lshiftc optimised for CPUs with fast SSE.

dnl  Contributed to the GNU project by David Harvey and Torbjorn Granlund.

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


C	     cycles/limb	     cycles/limb	      good
C          16-byte aligned         16-byte unaligned	    for cpu?
C AMD K8,K9	 ?			 ?
C AMD K10	 1.85  (1.635)		 1.9   (1.67)		Y
C AMD bd1	 1.82  (1.75)		 1.82  (1.75)		Y
C AMD bobcat	 4.5			 4.5
C Intel P4	 3.6   (3.125)		 3.6   (3.125)		Y
C Intel core2	 2.05  (1.67)		 2.55  (1.75)
C Intel NHM	 2.05  (1.875)		 2.6   (2.25)
C Intel SBR	 1.55  (1.44)		 2     (1.57)		Y
C Intel atom	 ?			 ?
C VIA nano	 2.5   (2.5)		 2.5   (2.5)		Y

C We try to do as many 16-byte operations as possible.  The top-most and
C bottom-most writes might need 8-byte operations.  We always write using
C 16-byte operations, we read with both 8-byte and 16-byte operations.

C There are two inner-loops, one for when rp = ap (mod 16) and one when this is
C not true.  The aligned case reads 16+8 bytes, the unaligned case reads
C 16+8+X bytes, where X is 8 or 16 depending on how punpcklqdq is implemented.

C This is not yet great code:
C   (1) The unaligned case makes too many reads.
C   (2) We should do some unrolling, at least 2-way.
C With 2-way unrolling but no scheduling we reach 1.5 c/l on K10 and 2 c/l on
C Nano.

C INPUT PARAMETERS
define(`rp',  `%rdi')
define(`ap',  `%rsi')
define(`n',   `%rdx')
define(`cnt', `%rcx')

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_lshiftc)
	movd	R32(%rcx), %xmm4
	mov	$64, R32(%rax)
	sub	R32(%rcx), R32(%rax)
	movd	R32(%rax), %xmm5

	neg	R32(%rcx)
	mov	-8(ap,n,8), %rax
	shr	R8(%rcx), %rax

	pcmpeqb	%xmm7, %xmm7		C set to 111...111

	cmp	$2, n
	jle	L(le2)

	lea	(rp,n,8), R32(%rcx)
	test	$8, R8(%rcx)
	je	L(rp_aligned)

C Do one initial limb in order to make rp aligned
	movq	-8(ap,n,8), %xmm0
	movq	-16(ap,n,8), %xmm1
	psllq	%xmm4, %xmm0
	psrlq	%xmm5, %xmm1
	por	%xmm1, %xmm0
	pxor	%xmm7, %xmm0
	movq	%xmm0, -8(rp,n,8)
	dec	n

L(rp_aligned):
	lea	(ap,n,8), R32(%rcx)
	test	$8, R8(%rcx)
	je	L(aent)
	jmp	L(uent)
C *****************************************************************************

C Handle the case when ap != rp (mod 16).

	ALIGN(16)
L(utop):movq	(ap,n,8), %xmm1
	punpcklqdq  8(ap,n,8), %xmm1
	movdqa	-8(ap,n,8), %xmm0
	psllq	%xmm4, %xmm1
	psrlq	%xmm5, %xmm0
	por	%xmm1, %xmm0
	pxor	%xmm7, %xmm0
	movdqa	%xmm0, (rp,n,8)
L(uent):sub	$2, n
	ja	L(utop)

	jne	L(end8)

	movq	(ap), %xmm1
	pxor	%xmm0, %xmm0
	punpcklqdq  %xmm1, %xmm0
	punpcklqdq  8(ap), %xmm1
	psllq	%xmm4, %xmm1
	psrlq	%xmm5, %xmm0
	por	%xmm1, %xmm0
	pxor	%xmm7, %xmm0
	movdqa	%xmm0, (rp)
	ret
C *****************************************************************************

C Handle the case when ap = rp (mod 16).

	ALIGN(16)
L(atop):movdqa	(ap,n,8), %xmm0		C xmm0 = B*ap[n-1] + ap[n-2]
	movq	-8(ap,n,8), %xmm1	C xmm1 = ap[n-3]
	punpcklqdq  %xmm0, %xmm1	C xmm1 = B*ap[n-2] + ap[n-3]
	psllq	%xmm4, %xmm0
	psrlq	%xmm5, %xmm1
	por	%xmm1, %xmm0
	pxor	%xmm7, %xmm0
	movdqa	%xmm0, (rp,n,8)
L(aent):sub	$2, n
	ja	L(atop)

	jne	L(end8)

	movdqa	(ap), %xmm0
	pxor	%xmm1, %xmm1
	punpcklqdq  %xmm0, %xmm1
	psllq	%xmm4, %xmm0
	psrlq	%xmm5, %xmm1
	por	%xmm1, %xmm0
	pxor	%xmm7, %xmm0
	movdqa	%xmm0, (rp)
	ret
C *****************************************************************************

	ALIGN(16)
L(le2):	jne	L(end8)

	movq	8(ap), %xmm0
	movq	(ap), %xmm1
	psllq	%xmm4, %xmm0
	psrlq	%xmm5, %xmm1
	por	%xmm1, %xmm0
	pxor	%xmm7, %xmm0
	movq	%xmm0, 8(rp)

L(end8):movq	(ap), %xmm0
	psllq	%xmm4, %xmm0
	pxor	%xmm7, %xmm0
	movq	%xmm0, (rp)
	ret
EPILOGUE()
