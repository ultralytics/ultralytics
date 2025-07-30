dnl  x86-64 mpn_rsh1add_n/mpn_rsh1sub_n optimized for Pentium 4.

dnl  Contributed to the GNU project by Torbjorn Granlund.

dnl  Copyright 2007, 2008, 2010-2012 Free Software Foundation, Inc.

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
C AMD K8,K9	 4.13
C AMD K10	 4.13
C Intel P4	 5.70
C Intel core2	 4.75
C Intel corei	 5
C Intel atom	 8.75
C VIA nano	 5.25

C TODO
C  * Try to make this smaller, 746 bytes seem excessive for this 2nd class
C    function.  Less sw pipelining would help, and since we now probably
C    pipeline somewhat too deeply, it might not affect performance too much.
C  * A separate small-n loop might speed things as well as make things smaller.
C    That loop should be selected before pushing registers.

C INPUT PARAMETERS
define(`rp',	`%rdi')
define(`up',	`%rsi')
define(`vp',	`%rdx')
define(`n',	`%rcx')
define(`cy',	`%r8')

ifdef(`OPERATION_rsh1add_n', `
	define(ADDSUB,	      add)
	define(func,	      mpn_rsh1add_n)
	define(func_nc,	      mpn_rsh1add_nc)')
ifdef(`OPERATION_rsh1sub_n', `
	define(ADDSUB,	      sub)
	define(func,	      mpn_rsh1sub_n)
	define(func_nc,	      mpn_rsh1sub_nc)')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

MULFUNC_PROLOGUE(mpn_rsh1add_n mpn_rsh1add_nc mpn_rsh1sub_n mpn_rsh1sub_nc)

ASM_START()
	TEXT
PROLOGUE(func)
	FUNC_ENTRY(4)
	xor	%r8, %r8
IFDOS(`	jmp	L(ent)		')
EPILOGUE()
PROLOGUE(func_nc)
	FUNC_ENTRY(4)
IFDOS(`	mov	56(%rsp), %r8	')
L(ent):	push	%rbx
	push	%r12
	push	%r13
	push	%r14
	push	%r15

	mov	(vp), %r9
	mov	(up), %r15

	mov	R32(n), R32(%rax)
	and	$3, R32(%rax)
	jne	L(n00)

	mov	R32(%r8), R32(%rbx)	C n = 0, 4, 8, ...
	mov	8(up), %r10
	ADDSUB	%r9, %r15
	mov	8(vp), %r9
	setc	R8(%rax)
	ADDSUB	%rbx, %r15		C return bit
	jnc	1f
	mov	$1, R8(%rax)
1:	mov	16(up), %r12
	ADDSUB	%r9, %r10
	mov	16(vp), %r9
	setc	R8(%rbx)
	mov	%r15, %r13
	ADDSUB	%rax, %r10
	jnc	1f
	mov	$1, R8(%rbx)
1:	mov	24(up), %r11
	ADDSUB	%r9, %r12
	lea	32(up), up
	mov	24(vp), %r9
	lea	32(vp), vp
	setc	R8(%rax)
	mov	%r10, %r14
	shl	$63, %r10
	shr	%r13
	jmp	L(L00)

L(n00):	cmp	$2, R32(%rax)
	jnc	L(n01)
	xor	R32(%rbx), R32(%rbx)	C n = 1, 5, 9, ...
	lea	-24(rp), rp
	mov	R32(%r8), R32(%rax)
	dec	n
	jnz	L(gt1)
	ADDSUB	%r9, %r15
	setc	R8(%rbx)
	ADDSUB	%rax, %r15
	jnc	1f
	mov	$1, R8(%rbx)
1:	mov	%r15, %r14
	shl	$63, %rbx
	shr	%r14
	jmp	L(cj1)
L(gt1):	mov	8(up), %r8
	ADDSUB	%r9, %r15
	mov	8(vp), %r9
	setc	R8(%rbx)
	ADDSUB	%rax, %r15
	jnc	1f
	mov	$1, R8(%rbx)
1:	mov	16(up), %r10
	ADDSUB	%r9, %r8
	mov	16(vp), %r9
	setc	R8(%rax)
	mov	%r15, %r14
	ADDSUB	%rbx, %r8
	jnc	1f
	mov	$1, R8(%rax)
1:	mov	24(up), %r12
	ADDSUB	%r9, %r10
	mov	24(vp), %r9
	setc	R8(%rbx)
	mov	%r8, %r13
	shl	$63, %r8
	shr	%r14
	lea	8(up), up
	lea	8(vp), vp
	jmp	L(L01)

L(n01):	jne	L(n10)
	lea	-16(rp), rp		C n = 2, 6, 10, ...
	mov	R32(%r8), R32(%rbx)
	mov	8(up), %r11
	ADDSUB	%r9, %r15
	mov	8(vp), %r9
	setc	R8(%rax)
	ADDSUB	%rbx, %r15
	jnc	1f
	mov	$1, R8(%rax)
1:	sub	$2, n
	jnz	L(gt2)
	ADDSUB	%r9, %r11
	setc	R8(%rbx)
	mov	%r15, %r13
	ADDSUB	%rax, %r11
	jnc	1f
	mov	$1, R8(%rbx)
1:	mov	%r11, %r14
	shl	$63, %r11
	shr	%r13
	jmp	L(cj2)
L(gt2):	mov	16(up), %r8
	ADDSUB	%r9, %r11
	mov	16(vp), %r9
	setc	R8(%rbx)
	mov	%r15, %r13
	ADDSUB	%rax, %r11
	jnc	1f
	mov	$1, R8(%rbx)
1:	mov	24(up), %r10
	ADDSUB	%r9, %r8
	mov	24(vp), %r9
	setc	R8(%rax)
	mov	%r11, %r14
	shl	$63, %r11
	shr	%r13
	lea	16(up), up
	lea	16(vp), vp
	jmp	L(L10)

L(n10):	xor	R32(%rbx), R32(%rbx)	C n = 3, 7, 11, ...
	lea	-8(rp), rp
	mov	R32(%r8), R32(%rax)
	mov	8(up), %r12
	ADDSUB	%r9, %r15
	mov	8(vp), %r9
	setc	R8(%rbx)
	ADDSUB	%rax, %r15
	jnc	1f
	mov	$1, R8(%rbx)
1:	mov	16(up), %r11
	ADDSUB	%r9, %r12
	mov	16(vp), %r9
	setc	R8(%rax)
	mov	%r15, %r14
	ADDSUB	%rbx, %r12
	jnc	1f
	mov	$1, R8(%rax)
1:	sub	$3, n
	jnz	L(gt3)
	ADDSUB	%r9, %r11
	setc	R8(%rbx)
	mov	%r12, %r13
	shl	$63, %r12
	shr	%r14
	jmp	L(cj3)
L(gt3):	mov	24(up), %r8
	ADDSUB	%r9, %r11
	mov	24(vp), %r9
	setc	R8(%rbx)
	mov	%r12, %r13
	shl	$63, %r12
	shr	%r14
	lea	24(up), up
	lea	24(vp), vp
	jmp	L(L11)

L(c0):	mov	$1, R8(%rbx)
	jmp	L(rc0)
L(c1):	mov	$1, R8(%rax)
	jmp	L(rc1)
L(c2):	mov	$1, R8(%rbx)
	jmp	L(rc2)

	ALIGN(16)
L(top):	mov	(up), %r8	C not on critical path
	or	%r13, %r10
	ADDSUB	%r9, %r11	C not on critical path
	mov	(vp), %r9	C not on critical path
	setc	R8(%rbx)	C save carry out
	mov	%r12, %r13	C new for later
	shl	$63, %r12	C shift new right
	shr	%r14		C shift old left
	mov	%r10, (rp)
L(L11):	ADDSUB	%rax, %r11	C apply previous carry out
	jc	L(c0)		C jump if ripple
L(rc0):	mov	8(up), %r10
	or	%r14, %r12
	ADDSUB	%r9, %r8
	mov	8(vp), %r9
	setc	R8(%rax)
	mov	%r11, %r14
	shl	$63, %r11
	shr	%r13
	mov	%r12, 8(rp)
L(L10):	ADDSUB	%rbx, %r8
	jc	L(c1)
L(rc1):	mov	16(up), %r12
	or	%r13, %r11
	ADDSUB	%r9, %r10
	mov	16(vp), %r9
	setc	R8(%rbx)
	mov	%r8, %r13
	shl	$63, %r8
	shr	%r14
	mov	%r11, 16(rp)
L(L01):	ADDSUB	%rax, %r10
	jc	L(c2)
L(rc2):	mov	24(up), %r11
	or	%r14, %r8
	ADDSUB	%r9, %r12
	lea	32(up), up
	mov	24(vp), %r9
	lea	32(vp), vp
	setc	R8(%rax)
	mov	%r10, %r14
	shl	$63, %r10
	shr	%r13
	mov	%r8, 24(rp)
	lea	32(rp), rp
L(L00):	ADDSUB	%rbx, %r12
	jc	L(c3)
L(rc3):	sub	$4, n
	ja	L(top)

L(end):	or	%r13, %r10
	ADDSUB	%r9, %r11
	setc	R8(%rbx)
	mov	%r12, %r13
	shl	$63, %r12
	shr	%r14
	mov	%r10, (rp)
L(cj3):	ADDSUB	%rax, %r11
	jnc	1f
	mov	$1, R8(%rbx)
1:	or	%r14, %r12
	mov	%r11, %r14
	shl	$63, %r11
	shr	%r13
	mov	%r12, 8(rp)
L(cj2):	or	%r13, %r11
	shl	$63, %rbx
	shr	%r14
	mov	%r11, 16(rp)
L(cj1):	or	%r14, %rbx
	mov	%rbx, 24(rp)

	mov	R32(%r15), R32(%rax)
	and	$1, R32(%rax)
	pop	%r15
	pop	%r14
	pop	%r13
	pop	%r12
	pop	%rbx
	FUNC_EXIT()
	ret
L(c3):	mov	$1, R8(%rax)
	jmp	L(rc3)
EPILOGUE()
