dnl  AMD64 mpn_addlshC_n, mpn_sublshC_n -- rp[] = up[] +- (vp[] << C), where
dnl  C is 1, 2, 3.  Optimized for Pentium 4.

dnl  Contributed to the GNU project by Torbjorn Granlund.

dnl  Copyright 2008, 2010-2012 Free Software Foundation, Inc.

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

C	     cycles/limb
C AMD K8,K9	 3.8
C AMD K10	 3.8
C Intel P4	 5.8
C Intel core2	 4.75
C Intel corei	 4.75
C Intel atom	 ?
C VIA nano	 4.75


C INPUT PARAMETERS
define(`rp',`%rdi')
define(`up',`%rsi')
define(`vp',`%rdx')
define(`n', `%rcx')

define(M, eval(m4_lshift(1,LSH)))

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(func)
	FUNC_ENTRY(4)
	push	%rbx
	push	%r12
	push	%rbp

	mov	(vp), %r9
	shl	$LSH, %r9
	mov	4(vp), R32(%rbp)

	xor	R32(%rbx), R32(%rbx)

	mov	R32(n), R32(%rax)
	and	$3, R32(%rax)
	jne	L(n00)		C n = 0, 4, 8, ...

	mov	(up), %r8
	mov	8(up), %r10
	shr	$RSH, R32(%rbp)
	ADDSUB	%r9, %r8
	mov	8(vp), %r9
	lea	(%rbp,%r9,M), %r9
	setc	R8(%rax)
	mov	12(vp), R32(%rbp)
	lea	-16(rp), rp
	jmp	L(L00)

L(n00):	cmp	$2, R32(%rax)
	jnc	L(n01)		C n = 1, 5, 9, ...
	mov	(up), %r11
	lea	-8(rp), rp
	shr	$RSH, R32(%rbp)
	ADDSUB	%r9, %r11
	setc	R8(%rbx)
	dec	n
	jz	L(1)		C jump for n = 1
	mov	8(up), %r8
	mov	8(vp), %r9
	lea	(%rbp,%r9,M), %r9
	mov	12(vp), R32(%rbp)
	lea	8(up), up
	lea	8(vp), vp
	jmp	L(L01)

L(n01):	jne	L(n10)		C n = 2, 6, 10, ...
	mov	(up), %r12
	mov	8(up), %r11
	shr	$RSH, R32(%rbp)
	ADDSUB	%r9, %r12
	mov	8(vp), %r9
	lea	(%rbp,%r9,M), %r9
	setc	R8(%rax)
	mov	12(vp), R32(%rbp)
	lea	16(up), up
	lea	16(vp), vp
	jmp	L(L10)

L(n10):	mov	(up), %r10
	mov	8(up), %r12
	shr	$RSH, R32(%rbp)
	ADDSUB	%r9, %r10
	mov	8(vp), %r9
	lea	(%rbp,%r9,M), %r9
	setc	R8(%rbx)
	mov	12(vp), R32(%rbp)
	lea	-24(rp), rp
	lea	-8(up), up
	lea	-8(vp), vp
	jmp	L(L11)

L(c0):	mov	$1, R8(%rbx)
	jmp	L(rc0)
L(c1):	mov	$1, R8(%rax)
	jmp	L(rc1)
L(c2):	mov	$1, R8(%rbx)
	jmp	L(rc2)

	ALIGN(16)
L(top):	mov	(up), %r8	C not on critical path
	shr	$RSH, R32(%rbp)
	ADDSUB	%r9, %r11	C not on critical path
	mov	(vp), %r9
	lea	(%rbp,%r9,M), %r9
	setc	R8(%rbx)	C save carry out
	mov	4(vp), R32(%rbp)
	mov	%r12, (rp)
	ADDSUB	%rax, %r11	C apply previous carry out
	jc	L(c0)		C jump if ripple
L(rc0):
L(L01):	mov	8(up), %r10
	shr	$RSH, R32(%rbp)
	ADDSUB	%r9, %r8
	mov	8(vp), %r9
	lea	(%rbp,%r9,M), %r9
	setc	R8(%rax)
	mov	12(vp), R32(%rbp)
	mov	%r11, 8(rp)
	ADDSUB	%rbx, %r8
	jc	L(c1)
L(rc1):
L(L00):	mov	16(up), %r12
	shr	$RSH, R32(%rbp)
	ADDSUB	%r9, %r10
	mov	16(vp), %r9
	lea	(%rbp,%r9,M), %r9
	setc	R8(%rbx)
	mov	20(vp), R32(%rbp)
	mov	%r8, 16(rp)
	ADDSUB	%rax, %r10
	jc	L(c2)
L(rc2):
L(L11):	mov	24(up), %r11
	shr	$RSH, R32(%rbp)
	ADDSUB	%r9, %r12
	mov	24(vp), %r9
	lea	(%rbp,%r9,M), %r9
	lea	32(up), up
	lea	32(vp), vp
	setc	R8(%rax)
	mov	-4(vp), R32(%rbp)
	mov	%r10, 24(rp)
	ADDSUB	%rbx, %r12
	jc	L(c3)
L(rc3):	lea	32(rp), rp
L(L10):	sub	$4, n
	ja	L(top)

L(end):
	shr	$RSH, R32(%rbp)
	ADDSUB	%r9, %r11
	setc	R8(%rbx)
	mov	%r12, (rp)
	ADDSUB	%rax, %r11
	jnc	L(1)
	mov	$1, R8(%rbx)
L(1):	mov	%r11, 8(rp)
	lea	(%rbx,%rbp), R32(%rax)
	pop	%rbp
	pop	%r12
	pop	%rbx
	FUNC_EXIT()
	ret
L(c3):	mov	$1, R8(%rax)
	jmp	L(rc3)
EPILOGUE()
ASM_END()
