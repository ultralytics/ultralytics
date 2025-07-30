dnl  AMD64 mpn_popcount -- population count.

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

include(`../config.m4')

C		    cycles/limb
C AMD K8,K9		 n/a
C AMD K10		 1.125
C Intel P4		 n/a
C Intel core2		 n/a
C Intel corei		 1.25
C Intel atom		 n/a
C VIA nano		 n/a

C * The zero-offset of popcount is misassembled to the offset-less form, which
C   is one byte shorter and therefore will mess up the switching code.
C * The outdated gas used in FreeBSD and NetBSD cannot handle the POPCNT insn,
C   which is the main reason for our usage of '.byte'.

C TODO
C  * Improve switching code, the current code sucks.

define(`up',		`%rdi')
define(`n',		`%rsi')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(32)
PROLOGUE(mpn_popcount)
	FUNC_ENTRY(2)

ifelse(1,1,`
	lea	(up,n,8), up

C	mov	R32(n), R32(%rcx)
C	neg	R32(%rcx)
	imul	$-1, R32(n), R32(%rcx)
	and	$8-1, R32(%rcx)

	neg	n

	mov	R32(%rcx), R32(%rax)
	neg	%rax
	lea	(up,%rax,8),up

	xor	R32(%rax), R32(%rax)

	lea	(%rcx,%rcx,4), %rcx

	lea	L(top)(%rip), %rdx
	lea	(%rdx,%rcx,2), %rdx
	jmp	*%rdx
',`
	lea	(up,n,8), up

	mov	R32(n), R32(%rcx)
	neg	R32(%rcx)
	and	$8-1, R32(%rcx)

	neg	n

	mov	R32(%rcx), R32(%rax)
	shl	$3, R32(%rax)
	sub	%rax, up

	xor	R32(%rax), R32(%rax)

C	add	R32(%rcx), R32(%rcx)	C 2x
C	lea	(%rcx,%rcx,4), %rcx	C 10x
	imul	$10, R32(%rcx)

	lea	L(top)(%rip), %rdx
	add	%rcx, %rdx
	jmp	*%rdx
')

	ALIGN(32)
L(top):
C 0 = n mod 8
	.byte	0xf3,0x4c,0x0f,0xb8,0x44,0xf7,0x00	C popcnt 0(up,n,8), %r8
	add	%r8, %rax
C 7 = n mod 8
	.byte	0xf3,0x4c,0x0f,0xb8,0x4c,0xf7,0x08	C popcnt 8(up,n,8), %r9
	add	%r9, %rax
C 6 = n mod 8
	.byte	0xf3,0x4c,0x0f,0xb8,0x44,0xf7,0x10	C popcnt 16(up,n,8), %r8
	add	%r8, %rax
C 5 = n mod 8
	.byte	0xf3,0x4c,0x0f,0xb8,0x4c,0xf7,0x18	C popcnt 24(up,n,8), %r9
	add	%r9, %rax
C 4 = n mod 8
	.byte	0xf3,0x4c,0x0f,0xb8,0x44,0xf7,0x20	C popcnt 32(up,n,8), %r8
	add	%r8, %rax
C 3 = n mod 8
	.byte	0xf3,0x4c,0x0f,0xb8,0x4c,0xf7,0x28	C popcnt 40(up,n,8), %r9
	add	%r9, %rax
C 2 = n mod 8
	.byte	0xf3,0x4c,0x0f,0xb8,0x44,0xf7,0x30	C popcnt 48(up,n,8), %r8
	add	%r8, %rax
C 1 = n mod 8
	.byte	0xf3,0x4c,0x0f,0xb8,0x4c,0xf7,0x38	C popcnt 56(up,n,8), %r9
	add	%r9, %rax

	add	$8, n
	js	L(top)
	FUNC_EXIT()
	ret
EPILOGUE()
