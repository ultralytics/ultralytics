dnl  x86-64 mpn_lshiftc optimized for "Core 2".

dnl  Copyright 2007, 2009, 2011, 2012 Free Software Foundation, Inc.

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


C	     cycles/limb
C AMD K8,K9	 ?
C AMD K10	 ?
C Intel P4	 ?
C Intel core2	 1.5
C Intel NHM	 2.25	(up to about n = 260, then 1.875)
C Intel SBR	 2.25
C Intel atom	 ?
C VIA nano	 ?


C INPUT PARAMETERS
define(`rp',	`%rdi')
define(`up',	`%rsi')
define(`n',	`%rdx')
define(`cnt',	`%rcx')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_lshiftc)
	FUNC_ENTRY(4)
	lea	-8(rp,n,8), rp
	lea	-8(up,n,8), up

	mov	R32(%rdx), R32(%rax)
	and	$3, R32(%rax)
	jne	L(nb00)
L(b00):	C n = 4, 8, 12, ...
	mov	(up), %r10
	mov	-8(up), %r11
	xor	R32(%rax), R32(%rax)
	shld	R8(cnt), %r10, %rax
	mov	-16(up), %r8
	lea	24(rp), rp
	sub	$4, n
	jmp	L(00)

L(nb00):C n = 1, 5, 9, ...
	cmp	$2, R32(%rax)
	jae	L(nb01)
L(b01):	mov	(up), %r9
	xor	R32(%rax), R32(%rax)
	shld	R8(cnt), %r9, %rax
	sub	$2, n
	jb	L(le1)
	mov	-8(up), %r10
	mov	-16(up), %r11
	lea	-8(up), up
	lea	16(rp), rp
	jmp	L(01)
L(le1):	shl	R8(cnt), %r9
	not	%r9
	mov	%r9, (rp)
	FUNC_EXIT()
	ret

L(nb01):C n = 2, 6, 10, ...
	jne	L(b11)
L(b10):	mov	(up), %r8
	mov	-8(up), %r9
	xor	R32(%rax), R32(%rax)
	shld	R8(cnt), %r8, %rax
	sub	$3, n
	jb	L(le2)
	mov	-16(up), %r10
	lea	-16(up), up
	lea	8(rp), rp
	jmp	L(10)
L(le2):	shld	R8(cnt), %r9, %r8
	not	%r8
	mov	%r8, (rp)
	shl	R8(cnt), %r9
	not	%r9
	mov	%r9, -8(rp)
	FUNC_EXIT()
	ret

	ALIGN(16)			C performance critical!
L(b11):	C n = 3, 7, 11, ...
	mov	(up), %r11
	mov	-8(up), %r8
	xor	R32(%rax), R32(%rax)
	shld	R8(cnt), %r11, %rax
	mov	-16(up), %r9
	lea	-24(up), up
	sub	$4, n
	jb	L(end)

	ALIGN(16)
L(top):	shld	R8(cnt), %r8, %r11
	mov	(up), %r10
	not	%r11
	mov	%r11, (rp)
L(10):	shld	R8(cnt), %r9, %r8
	mov	-8(up), %r11
	not	%r8
	mov	%r8, -8(rp)
L(01):	shld	R8(cnt), %r10, %r9
	mov	-16(up), %r8
	not	%r9
	mov	%r9, -16(rp)
L(00):	shld	R8(cnt), %r11, %r10
	mov	-24(up), %r9
	not	%r10
	mov	%r10, -24(rp)
	add	$-32, up
	lea	-32(rp), rp
	sub	$4, n
	jnc	L(top)

L(end):	shld	R8(cnt), %r8, %r11
	not	%r11
	mov	%r11, (rp)
	shld	R8(cnt), %r9, %r8
	not	%r8
	mov	%r8, -8(rp)
	shl	R8(cnt), %r9
	not	%r9
	mov	%r9, -16(rp)
	FUNC_EXIT()
	ret
EPILOGUE()
