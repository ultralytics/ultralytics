dnl  AMD64 mpn_popcount -- population count.

dnl  Copyright 2008, 2010-2013 Free Software Foundation, Inc.

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
C AMD K10		 1.5		slower than 8-way non-pipelined code
C AMD bd1		 4.2
C AMD bobcat		 6.28		slower than 8-way non-pipelined code
C Intel P4		 n/a
C Intel core2		 n/a
C Intel NHM		 1.32
C Intel SBR		 1.05		fluctuating
C Intel IBR		 1.05		fluctuating
C Intel HSW		 1
C Intel atom		 n/a
C VIA nano		 n/a

define(`up',		`%rdi')
define(`n_param',	`%rsi')

define(`n',		`%rcx')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(32)
PROLOGUE(mpn_popcount)
	FUNC_ENTRY(2)

	lea	(up,n_param,8), up
	xor	R32(%rax), R32(%rax)

	test	$1, R8(n_param)
	jnz	L(bx1)

L(bx0):	test	$2, R8(n_param)
	jnz	L(b10)

L(b00):	mov	$0, R32(n)
	sub	n_param, n
	.byte	0xf3,0x4c,0x0f,0xb8,0x04,0xcf		C popcnt (up,n,8), %r8
	.byte	0xf3,0x4c,0x0f,0xb8,0x4c,0xcf,0x08	C popcnt 8(up,n,8), %r9
	jmp	L(lo0)

L(b10):	mov	$2, R32(n)
	sub	n_param, n
	.byte	0xf3,0x4c,0x0f,0xb8,0x54,0xcf,0xf0	C popcnt -16(up,n,8), %r10
	.byte	0xf3,0x4c,0x0f,0xb8,0x5c,0xcf,0xf8	C popcnt -8(up,n,8), %r11
	test	n, n
	jz	L(cj2)
	jmp	L(lo2)

L(bx1):	test	$2, R8(n_param)
	jnz	L(b11)

L(b01):	mov	$1, R32(n)
	sub	n_param, n
	.byte	0xf3,0x4c,0x0f,0xb8,0x5c,0xcf,0xf8	C popcnt -8(up,n,8), %r11
	test	n, n
	jz	L(cj1)
	.byte	0xf3,0x4c,0x0f,0xb8,0x04,0xcf		C popcnt 0(up,n,8), %r8
	jmp	L(lo1)

L(b11):	mov	$-1, n
	sub	n_param, n
	.byte	0xf3,0x4c,0x0f,0xb8,0x4c,0xcf,0x08	C popcnt 8(up,n,8), %r9
	.byte	0xf3,0x4c,0x0f,0xb8,0x54,0xcf,0x10	C popcnt 16(up,n,8), %r10
	jmp	L(lo3)

	ALIGN(32)
L(top):	add	%r9, %rax
L(lo2):	.byte	0xf3,0x4c,0x0f,0xb8,0x04,0xcf		C popcnt 0(up,n,8), %r8
	add	%r10, %rax
L(lo1):	.byte	0xf3,0x4c,0x0f,0xb8,0x4c,0xcf,0x08	C popcnt 8(up,n,8), %r9
	add	%r11, %rax
L(lo0):	.byte	0xf3,0x4c,0x0f,0xb8,0x54,0xcf,0x10	C popcnt 16(up,n,8), %r10
	add	%r8, %rax
L(lo3):	.byte	0xf3,0x4c,0x0f,0xb8,0x5c,0xcf,0x18	C popcnt 24(up,n,8), %r11
	add	$4, n
	js	L(top)

L(end):	add	%r9, %rax
L(cj2):	add	%r10, %rax
L(cj1):	add	%r11, %rax
	FUNC_EXIT()
	ret
EPILOGUE()
