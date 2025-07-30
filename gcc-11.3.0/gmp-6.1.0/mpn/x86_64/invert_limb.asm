dnl  AMD64 mpn_invert_limb -- Invert a normalized limb.

dnl  Contributed to the GNU project by Torbjorn Granlund and Niels Möller.

dnl  Copyright 2004, 2007-2009, 2011, 2012 Free Software Foundation, Inc.

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


C	     cycles/limb (approx)	div
C AMD K8,K9	 48			 71
C AMD K10	 48			 77
C Intel P4	135			161
C Intel core2	 69			116
C Intel corei	 55			 89
C Intel atom	129			191
C VIA nano	 79			157

C rax rcx rdx rdi rsi r8

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

PROTECT(`mpn_invert_limb_table')

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_invert_limb)		C			Kn	C2	Ci
	FUNC_ENTRY(1)
	mov	%rdi, %rax		C			 0	 0	 0
	shr	$55, %rax		C			 1	 1	 1
ifdef(`PIC',`
ifdef(`DARWIN',`
	mov	mpn_invert_limb_table@GOTPCREL(%rip), %r8
	add	$-512, %r8
',`
	lea	-512+mpn_invert_limb_table(%rip), %r8
')',`
	movabs	$-512+mpn_invert_limb_table, %r8
')
	movzwl	(%r8,%rax,2), R32(%rcx)	C	%rcx = v0

	C v1 = (v0 << 11) - (v0*v0*d40 >> 40) - 1
	mov	%rdi, %rsi		C			 0	 0	 0
	mov	R32(%rcx), R32(%rax)	C			 4	 5	 5
	imul	R32(%rcx), R32(%rcx)	C			 4	 5	 5
	shr	$24, %rsi		C			 1	 1	 1
	inc	%rsi			C	%rsi = d40
	imul	%rsi, %rcx		C			 8	10	 8
	shr	$40, %rcx		C			12	15	11
	sal	$11, R32(%rax)		C			 5	 6	 6
	dec	R32(%rax)
	sub	R32(%rcx), R32(%rax)	C	%rax = v1

	C v2 = (v1 << 13) + (v1 * (2^60 - v1*d40) >> 47)
	mov	$0x1000000000000000, %rcx
	imul	%rax, %rsi		C			14	17	13
	sub	%rsi, %rcx
	imul	%rax, %rcx
	sal	$13, %rax
	shr	$47, %rcx
	add	%rax, %rcx		C	%rcx = v2

	C v3 = (v2 << 31) + (v2 * (2^96 - v2 * d63 + ((v2 >> 1) & mask)) >> 65
	mov	%rdi, %rsi		C			 0	 0	 0
	shr	%rsi			C d/2
	sbb	%rax, %rax		C -d0 = -(d mod 2)
	sub	%rax, %rsi		C d63 = ceil(d/2)
	imul	%rcx, %rsi		C v2 * d63
	and	%rcx, %rax		C v2 * d0
	shr	%rax			C (v2>>1) * d0
	sub	%rsi, %rax		C (v2>>1) * d0 - v2 * d63
	mul	%rcx
	sal	$31, %rcx
	shr	%rdx
	add	%rdx, %rcx		C	%rcx = v3

	mov	%rdi, %rax
	mul	%rcx
	add	%rdi, %rax
	mov	%rcx, %rax
	adc	%rdi, %rdx
	sub	%rdx, %rax

	FUNC_EXIT()
	ret
EPILOGUE()
ASM_END()
