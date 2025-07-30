dnl  AMD64 mpn_popcount, mpn_hamdist -- population count and hamming distance.

dnl  Copyright 2004, 2005, 2007, 2010-2012 Free Software Foundation, Inc.

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


C		     popcount	      hamdist
C		    cycles/limb	    cycles/limb
C AMD K8,K9		 6		 7
C AMD K10		 6		 7
C Intel P4		12		14.3
C Intel core2		 7		 8
C Intel corei		 ?		 7.3
C Intel atom		16.5		17.5
C VIA nano		 8.75		10.4

C TODO
C  * Tune.  It should be possible to reach 5 c/l for popcount and 6 c/l for
C    hamdist for K8/K9.


ifdef(`OPERATION_popcount',`
  define(`func',`mpn_popcount')
  define(`up',		`%rdi')
  define(`n',		`%rsi')
  define(`h55555555',	`%r10')
  define(`h33333333',	`%r11')
  define(`h0f0f0f0f',	`%rcx')
  define(`h01010101',	`%rdx')
  define(`POP',		`$1')
  define(`HAM',		`dnl')
')
ifdef(`OPERATION_hamdist',`
  define(`func',`mpn_hamdist')
  define(`up',		`%rdi')
  define(`vp',		`%rsi')
  define(`n',		`%rdx')
  define(`h55555555',	`%r10')
  define(`h33333333',	`%r11')
  define(`h0f0f0f0f',	`%rcx')
  define(`h01010101',	`%r14')
  define(`POP',		`dnl')
  define(`HAM',		`$1')
')


MULFUNC_PROLOGUE(mpn_popcount mpn_hamdist)

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(32)
PROLOGUE(func)
 POP(`	FUNC_ENTRY(2)		')
 HAM(`	FUNC_ENTRY(3)		')
	push	%r12
	push	%r13
 HAM(`	push	%r14		')

	mov	$0x5555555555555555, h55555555
	mov	$0x3333333333333333, h33333333
	mov	$0x0f0f0f0f0f0f0f0f, h0f0f0f0f
	mov	$0x0101010101010101, h01010101

	lea	(up,n,8), up
 HAM(`	lea	(vp,n,8), vp	')
	neg	n

	xor	R32(%rax), R32(%rax)

	bt	$0, R32(n)
	jnc	L(top)

	mov	(up,n,8), %r8
 HAM(`	xor	(vp,n,8), %r8	')

	mov	%r8, %r9
	shr	%r8
	and	h55555555, %r8
	sub	%r8, %r9

	mov	%r9, %r8
	shr	$2, %r9
	and	h33333333, %r8
	and	h33333333, %r9
	add	%r8, %r9		C 16 4-bit fields (0..4)

	mov	%r9, %r8
	shr	$4, %r9
	and	h0f0f0f0f, %r8
	and	h0f0f0f0f, %r9
	add	%r8, %r9		C 8 8-bit fields (0..16)

	imul	h01010101, %r9		C sum the 8 fields in high 8 bits
	shr	$56, %r9

	mov	%r9, %rax		C add to total
	add	$1, n
	jz	L(end)

	ALIGN(16)
L(top):	mov	(up,n,8), %r8
	mov	8(up,n,8), %r12
 HAM(`	xor	(vp,n,8), %r8	')
 HAM(`	xor	8(vp,n,8), %r12	')

	mov	%r8, %r9
	mov	%r12, %r13
	shr	%r8
	shr	%r12
	and	h55555555, %r8
	and	h55555555, %r12
	sub	%r8, %r9
	sub	%r12, %r13

	mov	%r9, %r8
	mov	%r13, %r12
	shr	$2, %r9
	shr	$2, %r13
	and	h33333333, %r8
	and	h33333333, %r9
	and	h33333333, %r12
	and	h33333333, %r13
	add	%r8, %r9		C 16 4-bit fields (0..4)
	add	%r12, %r13		C 16 4-bit fields (0..4)

	add	%r13, %r9		C 16 4-bit fields (0..8)
	mov	%r9, %r8
	shr	$4, %r9
	and	h0f0f0f0f, %r8
	and	h0f0f0f0f, %r9
	add	%r8, %r9		C 8 8-bit fields (0..16)

	imul	h01010101, %r9		C sum the 8 fields in high 8 bits
	shr	$56, %r9

	add	%r9, %rax		C add to total
	add	$2, n
	jnc	L(top)

L(end):
 HAM(`	pop	%r14		')
	pop	%r13
	pop	%r12
	FUNC_EXIT()
	ret
EPILOGUE()
