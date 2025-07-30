dnl  AMD64 mpn_com optimised for CPUs with fast SSE.

dnl  Copyright 2003, 2005, 2007, 2011, 2012, 2015 Free Software Foundation,
dnl  Inc.

dnl  Contributed to the GNU project by Torbjorn Granlund.

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

C	     cycles/limb     cycles/limb     cycles/limb      good
C              aligned	      unaligned	      best seen	     for cpu?
C AMD K8,K9	 2.0		 2.0				N
C AMD K10	 0.85		 1.3				Y/N
C AMD bull	 1.40		 1.40				Y
C AMD pile     0.9-1.4	       0.9-1.4				Y
C AMD steam
C AMD excavator
C AMD bobcat	 3.1		 3.1				N
C AMD jaguar	 0.91		 0.91		opt/opt		Y
C Intel P4	 2.28		 illop				Y
C Intel core2	 1.02		 1.02				N
C Intel NHM	 0.53		 0.68				Y
C Intel SBR	 0.51		 0.75		opt/0.65	Y/N
C Intel IBR	 0.50		 0.57		opt/opt		Y
C Intel HWL	 0.51		 0.64		opt/0.58	Y
C Intel BWL	 0.61		 0.65		0.57/opt	Y
C Intel atom	 3.68		 3.68				N
C Intel SLM	 1.09		 1.35				N
C VIA nano	 1.17		 5.09				Y/N

C We try to do as many 16-byte operations as possible.  The top-most and
C bottom-most writes might need 8-byte operations.  We can always write using
C aligned 16-byte operations, we read with both aligned and unaligned 16-byte
C operations.

C Instead of having separate loops for reading aligned and unaligned, we read
C using MOVDQU.  This seems to work great except for core2; there performance
C doubles when reading using MOVDQA (for aligned source).  It is unclear how to
C best handle the unaligned case there.

C INPUT PARAMETERS
define(`rp', `%rdi')
define(`up', `%rsi')
define(`n',  `%rdx')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_com)
	FUNC_ENTRY(3)

	pcmpeqb	%xmm7, %xmm7		C set to 111...111

	test	$8, R8(rp)		C is rp 16-byte aligned?
	jz	L(ali)			C jump if rp aligned
	mov	(up), %rax
	lea	8(up), up
	not	%rax
	mov	%rax, (rp)
	lea	8(rp), rp
	dec	n

	sub	$14, n
	jc	L(sma)

	ALIGN(16)
L(top):	movdqu	(up), %xmm0
	movdqu	16(up), %xmm1
	movdqu	32(up), %xmm2
	movdqu	48(up), %xmm3
	movdqu	64(up), %xmm4
	movdqu	80(up), %xmm5
	movdqu	96(up), %xmm6
	lea	112(up), up
	pxor	%xmm7, %xmm0
	pxor	%xmm7, %xmm1
	pxor	%xmm7, %xmm2
	pxor	%xmm7, %xmm3
	pxor	%xmm7, %xmm4
	pxor	%xmm7, %xmm5
	pxor	%xmm7, %xmm6
	movdqa	%xmm0, (rp)
	movdqa	%xmm1, 16(rp)
	movdqa	%xmm2, 32(rp)
	movdqa	%xmm3, 48(rp)
	movdqa	%xmm4, 64(rp)
	movdqa	%xmm5, 80(rp)
	movdqa	%xmm6, 96(rp)
	lea	112(rp), rp
L(ali):	sub	$14, n
	jnc	L(top)

L(sma):	add	$14, n
	test	$8, R8(n)
	jz	1f
	movdqu	(up), %xmm0
	movdqu	16(up), %xmm1
	movdqu	32(up), %xmm2
	movdqu	48(up), %xmm3
	lea	64(up), up
	pxor	%xmm7, %xmm0
	pxor	%xmm7, %xmm1
	pxor	%xmm7, %xmm2
	pxor	%xmm7, %xmm3
	movdqa	%xmm0, (rp)
	movdqa	%xmm1, 16(rp)
	movdqa	%xmm2, 32(rp)
	movdqa	%xmm3, 48(rp)
	lea	64(rp), rp
1:
	test	$4, R8(n)
	jz	1f
	movdqu	(up), %xmm0
	movdqu	16(up), %xmm1
	lea	32(up), up
	pxor	%xmm7, %xmm0
	pxor	%xmm7, %xmm1
	movdqa	%xmm0, (rp)
	movdqa	%xmm1, 16(rp)
	lea	32(rp), rp
1:
	test	$2, R8(n)
	jz	1f
	movdqu	(up), %xmm0
	lea	16(up), up
	pxor	%xmm7, %xmm0
	movdqa	%xmm0, (rp)
	lea	16(rp), rp
1:
	test	$1, R8(n)
	jz	1f
	mov	(up), %rax
	not	%rax
	mov	%rax, (rp)
1:
L(don):	FUNC_EXIT()
	ret
EPILOGUE()
