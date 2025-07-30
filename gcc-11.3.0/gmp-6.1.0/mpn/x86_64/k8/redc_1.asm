dnl  X86-64 mpn_redc_1 optimised for AMD K8-K10.

dnl  Contributed to the GNU project by Torbjörn Granlund.

dnl  Copyright 2004, 2008, 2013 Free Software Foundation, Inc.

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
C AMD bull	 ?
C AMD pile	 ?
C AMD steam	 ?
C AMD bobcat	 ?
C AMD jaguar	 ?
C Intel P4	 ?
C Intel core	 ?
C Intel NHM	 ?
C Intel SBR	 ?
C Intel IBR	 ?
C Intel HWL	 ?
C Intel BWL	 ?
C Intel atom	 ?
C VIA nano	 ?

C The inner loops of this code are the result of running a code generation and
C optimisation tool suite written by David Harvey and Torbjörn Granlund.

C TODO
C  * Micro-optimise, none performed thus far.
C  * This looks different from other current redc_1.asm variants.  Consider
C    adapting this to the mainstream style.
C  * Is this code really faster than more approaches which compute q0 later?
C    Is the use of a jump jump table faster?  Or is the edge of this due to the
C    inlined add_n code?
C  * Put initial m[0] x q0 computation in header.
C  * Put basecases at the file's end, single them out before the pushes.

define(`rp',          `%rdi')   C rcx
define(`up',          `%rsi')   C rdx
define(`mp_param',    `%rdx')   C r8
define(`n',           `%rcx')   C r9
define(`u0inv',       `%r8')    C stack

define(`i',           `%r11')
define(`nneg',        `%r12')
define(`mp',          `%r13')
define(`q0',          `%rbp')
define(`vp',          `%rdx')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(32)
PROLOGUE(mpn_redc_1)
	FUNC_ENTRY(4)
IFDOS(`	mov	56(%rsp), %r8	')
	push	%rbp
	mov	(up), q0		C up[0]
	push	%rbx
	imul	u0inv, q0		C first q0, for all execution paths
	push	%r12
	push	%r13
	push	%r14
	push	%r15

	mov	n, nneg
	neg	nneg
	lea	(mp_param,n,8), mp	C mp += n
	lea	-16(up,n,8), up		C up += n

	mov	R32(n), R32(%rax)
	and	$3, R32(%rax)
	lea	4(%rax), %r9
	cmp	$4, R32(n)
	cmovg	%r9, %rax
	lea	L(tab)(%rip), %r9
ifdef(`PIC',`
	movslq	(%r9,%rax,4), %rax
	add	%r9, %rax
	jmp	*%rax
',`
	jmp	*(%r9,%rax,8)
')

	JUMPTABSECT
	ALIGN(8)
L(tab):	JMPENT(	L(0), L(tab))
	JMPENT(	L(1), L(tab))
	JMPENT(	L(2), L(tab))
	JMPENT(	L(3), L(tab))
	JMPENT(	L(0m4), L(tab))
	JMPENT(	L(1m4), L(tab))
	JMPENT(	L(2m4), L(tab))
	JMPENT(	L(3m4), L(tab))
	TEXT

	ALIGN(16)
L(1):	mov	(mp_param), %rax
	mul	q0
	add	8(up), %rax
	adc	16(up), %rdx
	mov	%rdx, (rp)
	mov	$0, R32(%rax)
	adc	R32(%rax), R32(%rax)
	jmp	L(ret)


	ALIGN(16)
L(2):	mov	(mp_param), %rax
	mul	q0
	xor	R32(%r14), R32(%r14)
	mov	%rax, %r10
	mov	-8(mp), %rax
	mov	%rdx, %r9
	mul	q0
	add	(up), %r10
	adc	%rax, %r9
	adc	%rdx, %r14
	add	8(up), %r9
	adc	$0, %r14
	mov	%r9, q0
	imul	u0inv, q0
	mov	-16(mp), %rax
	mul	q0
	xor	R32(%rbx), R32(%rbx)
	mov	%rax, %r10
	mov	-8(mp), %rax
	mov	%rdx, %r11
	mul	q0
	add	%r9, %r10
	adc	%rax, %r11
	adc	%rdx, %rbx
	add	16(up), %r11
	adc	$0, %rbx
	xor	R32(%rax), R32(%rax)
	add	%r11, %r14
	adc	24(up), %rbx
	mov	%r14, (rp)
	mov	%rbx, 8(rp)
	adc	R32(%rax), R32(%rax)
	jmp	L(ret)


L(3):	mov	(mp_param), %rax
	mul	q0
	mov	%rax, %rbx
	mov	%rdx, %r10
	mov	-16(mp), %rax
	mul	q0
	xor	R32(%r9), R32(%r9)
	xor	R32(%r14), R32(%r14)
	add	-8(up), %rbx
	adc	%rax, %r10
	mov	-8(mp), %rax
	adc	%rdx, %r9
	mul	q0
	add	(up), %r10
	mov	%r10, (up)
	adc	%rax, %r9
	adc	%rdx, %r14
	mov	%r10, q0
	imul	u0inv, q0
	add	%r9, 8(up)
	adc	$0, %r14
	mov	%r14, -8(up)

	mov	-24(mp), %rax
	mul	q0
	mov	%rax, %rbx
	mov	%rdx, %r10
	mov	-16(mp), %rax
	mul	q0
	xor	R32(%r9), R32(%r9)
	xor	R32(%r14), R32(%r14)
	add	(up), %rbx
	adc	%rax, %r10
	mov	-8(mp), %rax
	adc	%rdx, %r9
	mul	q0
	add	8(up), %r10
	mov	%r10, 8(up)
	adc	%rax, %r9
	adc	%rdx, %r14
	mov	%r10, q0
	imul	u0inv, q0
	add	%r9, 16(up)
	adc	$0, %r14
	mov	%r14, (up)

	mov	-24(mp), %rax
	mul	q0
	mov	%rax, %rbx
	mov	%rdx, %r10
	mov	-16(mp), %rax
	mul	q0
	xor	R32(%r9), R32(%r9)
	xor	R32(%r14), R32(%r14)
	add	8(up), %rbx
	adc	%rax, %r10
	mov	-8(mp), %rax
	adc	%rdx, %r9
	mul	q0
	add	16(up), %r10
	adc	%rax, %r9
	adc	%rdx, %r14
	add	24(up), %r9
	adc	$0, %r14

	xor	R32(%rax), R32(%rax)
	add	-8(up), %r10
	adc	(up), %r9
	adc	32(up), %r14
	mov	%r10, (rp)
	mov	%r9, 8(rp)
	mov	%r14, 16(rp)
	adc	R32(%rax), R32(%rax)
	jmp	L(ret)


	ALIGN(16)
L(2m4):
L(lo2):	mov	(mp,nneg,8), %rax
	mul	q0
	xor	R32(%r14), R32(%r14)
	xor	R32(%rbx), R32(%rbx)
	mov	%rax, %r10
	mov	8(mp,nneg,8), %rax
	mov	24(up,nneg,8), %r15
	mov	%rdx, %r9
	mul	q0
	add	16(up,nneg,8), %r10
	adc	%rax, %r9
	mov	16(mp,nneg,8), %rax
	adc	%rdx, %r14
	mul	q0
	mov	$0, R32(%r10)		C xor?
	lea	2(nneg), i
	add	%r9, %r15
	imul	u0inv, %r15
	jmp	 L(e2)

	ALIGN(16)
L(li2):	add	%r10, (up,i,8)
	adc	%rax, %r9
	mov	(mp,i,8), %rax
	adc	%rdx, %r14
	xor	R32(%r10), R32(%r10)
	mul	q0
L(e2):	add	%r9, 8(up,i,8)
	adc	%rax, %r14
	adc	%rdx, %rbx
	mov	8(mp,i,8), %rax
	mul	q0
	add	%r14, 16(up,i,8)
	adc	%rax, %rbx
	adc	%rdx, %r10
	mov	16(mp,i,8), %rax
	mul	q0
	add	%rbx, 24(up,i,8)
	mov	$0, R32(%r14)		C zero
	mov	%r14, %rbx		C zero
	adc	%rax, %r10
	mov	24(mp,i,8), %rax
	mov	%r14, %r9		C zero
	adc	%rdx, %r9
	mul	q0
	add	$4, i
	js	 L(li2)

L(le2):	add	%r10, (up)
	adc	%rax, %r9
	adc	%r14, %rdx
	add	%r9, 8(up)
	adc	$0, %rdx
	mov	%rdx, 16(up,nneg,8)	C up[0]
	add	$8, up
	mov	%r15, q0
	dec	n
	jnz	L(lo2)

	mov	nneg, n
	sar	$2, n
	lea	32(up,nneg,8), up
	lea	(up,nneg,8), vp

	mov	-16(up), %r8
	mov	-8(up), %r9
	add	-16(vp), %r8
	adc	-8(vp), %r9
	mov	%r8, (rp)
	mov	%r9, 8(rp)
	lea	16(rp), rp
	jmp	L(addx)


	ALIGN(16)
L(1m4):
L(lo1):	mov	(mp,nneg,8), %rax
	xor	%r9, %r9
	xor	R32(%rbx), R32(%rbx)
	mul	q0
	mov	%rax, %r9
	mov	8(mp,nneg,8), %rax
	mov	24(up,nneg,8), %r15
	mov	%rdx, %r14
	mov	$0, R32(%r10)		C xor?
	mul	q0
	add	16(up,nneg,8), %r9
	adc	%rax, %r14
	adc	%rdx, %rbx
	mov	16(mp,nneg,8), %rax
	mul	q0
	lea	1(nneg), i
	add	%r14, %r15
	imul	u0inv, %r15
	jmp	 L(e1)

	ALIGN(16)
L(li1):	add	%r10, (up,i,8)
	adc	%rax, %r9
	mov	(mp,i,8), %rax
	adc	%rdx, %r14
	xor	R32(%r10), R32(%r10)
	mul	q0
	add	%r9, 8(up,i,8)
	adc	%rax, %r14
	adc	%rdx, %rbx
	mov	8(mp,i,8), %rax
	mul	q0
L(e1):	add	%r14, 16(up,i,8)
	adc	%rax, %rbx
	adc	%rdx, %r10
	mov	16(mp,i,8), %rax
	mul	q0
	add	%rbx, 24(up,i,8)
	mov	$0, R32(%r14)		C zero
	mov	%r14, %rbx		C zero
	adc	%rax, %r10
	mov	24(mp,i,8), %rax
	mov	%r14, %r9		C zero
	adc	%rdx, %r9
	mul	q0
	add	$4, i
	js	 L(li1)

L(le1):	add	%r10, (up)
	adc	%rax, %r9
	adc	%r14, %rdx
	add	%r9, 8(up)
	adc	$0, %rdx
	mov	%rdx, 16(up,nneg,8)	C up[0]
	add	$8, up
	mov	%r15, q0
	dec	n
	jnz	L(lo1)

	mov	nneg, n
	sar	$2, n
	lea	24(up,nneg,8), up
	lea	(up,nneg,8), vp

	mov	-8(up), %r8
	add	-8(vp), %r8
	mov	%r8, (rp)
	lea	8(rp), rp
	jmp	L(addx)


	ALIGN(16)
L(0):
L(0m4):
L(lo0):	mov	(mp,nneg,8), %rax
	mov	nneg, i
	mul	q0
	xor	R32(%r10), R32(%r10)
	mov	%rax, %r14
	mov	%rdx, %rbx
	mov	8(mp,nneg,8), %rax
	mov	24(up,nneg,8), %r15
	mul	q0
	add	16(up,nneg,8), %r14
	adc	%rax, %rbx
	adc	%rdx, %r10
	add	%rbx, %r15
	imul	u0inv, %r15
	jmp	L(e0)

	ALIGN(16)
L(li0):	add	%r10, (up,i,8)
	adc	%rax, %r9
	mov	(mp,i,8), %rax
	adc	%rdx, %r14
	xor	R32(%r10), R32(%r10)
	mul	q0
	add	%r9, 8(up,i,8)
	adc	%rax, %r14
	adc	%rdx, %rbx
	mov	8(mp,i,8), %rax
	mul	q0
	add	%r14, 16(up,i,8)
	adc	%rax, %rbx
	adc	%rdx, %r10
L(e0):	mov	16(mp,i,8), %rax
	mul	q0
	add	%rbx, 24(up,i,8)
	mov	$0, R32(%r14)		C zero
	mov	%r14, %rbx		C zero
	adc	%rax, %r10
	mov	24(mp,i,8), %rax
	mov	%r14, %r9		C zero
	adc	%rdx, %r9
	mul	q0
	add	$4, i
	js	 L(li0)

L(le0):	add	%r10, (up)
	adc	%rax, %r9
	adc	%r14, %rdx
	add	%r9, 8(up)
	adc	$0, %rdx
	mov	%rdx, 16(up,nneg,8)	C up[0]
	add	$8, up
	mov	%r15, q0
	dec	n
	jnz	L(lo0)

	mov	nneg, n
	sar	$2, n
	clc
	lea	16(up,nneg,8), up
	lea	(up,nneg,8), vp
	jmp	L(addy)


	ALIGN(16)
L(3m4):
L(lo3):	mov	(mp,nneg,8), %rax
	mul	q0
	mov	%rax, %rbx
	mov	%rdx, %r10
	mov	8(mp,nneg,8), %rax
	mov	24(up,nneg,8), %r15
	mul	q0
	add	16(up,nneg,8), %rbx	C result is zero, might carry
	mov	$0, R32(%rbx)		C zero
	mov	%rbx, %r14		C zero
	adc	%rax, %r10
	mov	16(mp,nneg,8), %rax
	mov	%r14, %r9		C zero
	adc	%rdx, %r9
	add	%r10, %r15
	mul	q0
	lea	3(nneg), i
	imul	u0inv, %r15
C	jmp	L(li3)

	ALIGN(16)
L(li3):	add	%r10, (up,i,8)
	adc	%rax, %r9
	mov	(mp,i,8), %rax
	adc	%rdx, %r14
	xor	R32(%r10), R32(%r10)
	mul	q0
	add	%r9, 8(up,i,8)
	adc	%rax, %r14
	adc	%rdx, %rbx
	mov	8(mp,i,8), %rax
	mul	q0
	add	%r14, 16(up,i,8)
	adc	%rax, %rbx
	adc	%rdx, %r10
	mov	16(mp,i,8), %rax
	mul	q0
	add	%rbx, 24(up,i,8)
	mov	$0, R32(%r14)		C zero
	mov	%r14, %rbx		C zero
	adc	%rax, %r10
	mov	24(mp,i,8), %rax
	mov	%r14, %r9		C zero
	adc	%rdx, %r9
	mul	q0
	add	$4, i
	js	 L(li3)

L(le3):	add	%r10, (up)
	adc	%rax, %r9
	adc	%r14, %rdx
	add	%r9, 8(up)
	adc	$0, %rdx
	mov	%rdx, 16(up,nneg,8)	C up[0]
	mov	%r15, q0
	lea	8(up), up
	dec	n
	jnz	L(lo3)


C ==== Addition code ====
	mov	nneg, n
	sar	$2, n
	lea	40(up,nneg,8), up
	lea	(up,nneg,8), vp

	mov	-24(up), %r8
	mov	-16(up), %r9
	mov	-8(up), %r10
	add	-24(vp), %r8
	adc	-16(vp), %r9
	adc	-8(vp), %r10
	mov	%r8, (rp)
	mov	%r9, 8(rp)
	mov	%r10, 16(rp)
	lea	24(rp), rp

L(addx):inc	n
	jz	L(ad3)

L(addy):mov	(up), %r8
	mov	8(up), %r9
	inc	n
	jmp	L(mid)

C	ALIGN(16)
L(al3):	adc	(vp), %r8
	adc	8(vp), %r9
	adc	16(vp), %r10
	adc	24(vp), %r11
	mov	%r8, (rp)
	lea	32(up), up
	mov	%r9, 8(rp)
	mov	%r10, 16(rp)
	inc	n
	mov	%r11, 24(rp)
	lea	32(vp), vp
	mov	(up), %r8
	mov	8(up), %r9
	lea	32(rp), rp
L(mid):	mov	16(up), %r10
	mov	24(up), %r11
	jnz	L(al3)

L(ae3):	adc	(vp), %r8
	adc	8(vp), %r9
	adc	16(vp), %r10
	adc	24(vp), %r11
	mov	%r8, (rp)
	mov	%r9, 8(rp)
	mov	%r10, 16(rp)
	mov	%r11, 24(rp)

L(ad3):	mov	R32(n), R32(%rax)	C zero
	adc	R32(%rax), R32(%rax)

L(ret):	pop	%r15
	pop	%r14
	pop	%r13
	pop	%r12
	pop	%rbx
	pop	%rbp
	FUNC_EXIT()
	ret
EPILOGUE()
