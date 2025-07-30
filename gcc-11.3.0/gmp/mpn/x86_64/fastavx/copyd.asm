dnl  AMD64 mpn_copyd optimised for CPUs with fast AVX.

dnl  Copyright 2003, 2005, 2007, 2011-2013, 2015 Free Software Foundation, Inc.

dnl  Contributed to the GNU project by Torbjörn Granlund.

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
C AMD K8,K9	n/a
C AMD K10	n/a
C AMD bull	n/a
C AMD pile	 4.87		 4.87				N
C AMD steam	 ?		 ?
C AMD bobcat	n/a
C AMD jaguar	n/a
C Intel P4	n/a
C Intel core	n/a
C Intel NHM	n/a
C Intel SBR	 0.50		 0.91				N
C Intel IBR	 0.50		 0.65				N
C Intel HWL	 0.25		 0.30				Y
C Intel BWL	 0.28		 0.37				Y
C Intel atom	n/a
C VIA nano	n/a

C We try to do as many 32-byte operations as possible.  The top-most and
C bottom-most writes might need 8-byte operations.  For the bulk copying, we
C write using aligned 32-byte operations, but we read with both aligned and
C unaligned 32-byte operations.

define(`rp', `%rdi')
define(`up', `%rsi')
define(`n',  `%rdx')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

dnl define(`vmovdqu', vlddqu)

ASM_START()
	TEXT
	ALIGN(32)
PROLOGUE(mpn_copyd)
	FUNC_ENTRY(3)

	lea	-32(rp,n,8), rp
	lea	-32(up,n,8), up

	cmp	$7, n			C basecase needed for correctness
	jbe	L(bc)

	test	$8, R8(rp)		C is rp 16-byte aligned?
	jz	L(a2)			C jump if rp aligned
	mov	24(up), %rax
	lea	-8(up), up
	mov	%rax, 24(rp)
	lea	-8(rp), rp
	dec	n
L(a2):	test	$16, R8(rp)		C is rp 32-byte aligned?
	jz	L(a3)			C jump if rp aligned
	vmovdqu	16(up), %xmm0
	lea	-16(up), up
	vmovdqa	%xmm0, 16(rp)
	lea	-16(rp), rp
	sub	$2, n
L(a3):	sub	$16, n
	jc	L(sma)

	ALIGN(16)
L(top):	vmovdqu	(up), %ymm0
	vmovdqu	-32(up), %ymm1
	vmovdqu	-64(up), %ymm2
	vmovdqu	-96(up), %ymm3
	lea	-128(up), up
	vmovdqa	%ymm0, (rp)
	vmovdqa	%ymm1, -32(rp)
	vmovdqa	%ymm2, -64(rp)
	vmovdqa	%ymm3, -96(rp)
	lea	-128(rp), rp
L(ali):	sub	$16, n
	jnc	L(top)

L(sma):	test	$8, R8(n)
	jz	1f
	vmovdqu	(up), %ymm0
	vmovdqu	-32(up), %ymm1
	lea	-64(up), up
	vmovdqa	%ymm0, (rp)
	vmovdqa	%ymm1, -32(rp)
	lea	-64(rp), rp
1:
	test	$4, R8(n)
	jz	1f
	vmovdqu	(up), %ymm0
	lea	-32(up), up
	vmovdqa	%ymm0, (rp)
	lea	-32(rp), rp
1:
	test	$2, R8(n)
	jz	1f
	vmovdqu	16(up), %xmm0
	lea	-16(up), up
	vmovdqa	%xmm0, 16(rp)
	lea	-16(rp), rp
1:
	test	$1, R8(n)
	jz	1f
	mov	24(up), %r8
	mov	%r8, 24(rp)
1:
	FUNC_EXIT()
	ret

	ALIGN(16)
L(bc):	test	$4, R8(n)
	jz	1f
	mov	24(up), %rax
	mov	16(up), %rcx
	mov	8(up), %r8
	mov	(up), %r9
	lea	-32(up), up
	mov	%rax, 24(rp)
	mov	%rcx, 16(rp)
	mov	%r8, 8(rp)
	mov	%r9, (rp)
	lea	-32(rp), rp
1:
	test	$2, R8(n)
	jz	1f
	mov	24(up), %rax
	mov	16(up), %rcx
	lea	-16(up), up
	mov	%rax, 24(rp)
	mov	%rcx, 16(rp)
	lea	-16(rp), rp
1:
	test	$1, R8(n)
	jz	1f
	mov	24(up), %rax
	mov	%rax, 24(rp)
1:
	FUNC_EXIT()
	ret
EPILOGUE()
