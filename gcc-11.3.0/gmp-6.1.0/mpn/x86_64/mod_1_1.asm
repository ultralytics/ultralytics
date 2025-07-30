dnl  AMD64 mpn_mod_1_1p

dnl  Contributed to the GNU project by Torbjörn Granlund and Niels Möller.

dnl  Copyright 2009-2012, 2014 Free Software Foundation, Inc.

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
C AMD K8,K9	 6
C AMD K10	 6
C Intel P4	26
C Intel core2	12.5
C Intel NHM	11.3
C Intel SBR	 8.4	(slowdown, old code took 8.0)
C Intel atom	26
C VIA nano	13

define(`B2mb',   `%r10')
define(`B2modb', `%r11')
define(`ap',     `%rdi')
define(`n',      `%rsi')
define(`pre',    `%r8')
define(`b',      `%rbx')

define(`r0',     `%rbp') C r1 kept in %rax
define(`r2',	 `%rcx')  C kept negated. Also used as shift count
define(`t0',     `%r9')

C mp_limb_t
C mpn_mod_1_1p (mp_srcptr ap, mp_size_t n, mp_limb_t b, mp_limb_t bmodb[4])
C                       %rdi         %rsi         %rdx                %rcx
C The pre array contains bi, cnt, B1modb, B2modb
C Note: This implementation needs B1modb only when cnt > 0

C The iteration is almost as follows,
C
C   r_2 B^3 + r_1 B^2 + r_0 B + u = r_1 B2modb + (r_0 + r_2 B2mod) B + u
C
C where r2 is a single bit represented as a mask. But to make sure that the
C result fits in two limbs and a bit, carry from the addition
C
C   r_0 + r_2 B2mod
C
C is handled specially. On carry, we subtract b to cancel the carry,
C and we use instead the value
C
C   r_0 + B2mb (mod B)
C
C This addition can be issued early since it doesn't depend on r2, and it is
C the source of the cmov in the loop.
C
C We have the invariant that r_2 B^2 + r_1 B + r_0 < B^2 + B b

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_mod_1_1p)
	FUNC_ENTRY(4)
	push	%rbp
	push	%rbx
	mov	%rdx, b
	mov	%rcx, pre

	mov	-8(ap, n, 8), %rax
	cmp	$3, n
	jnc	L(first)
	mov	-16(ap, n, 8), r0
	jmp	L(reduce_two)

L(first):
	C First iteration, no r2
	mov	24(pre), B2modb
	mul	B2modb
	mov	-24(ap, n, 8), r0
	add	%rax, r0
	mov	-16(ap, n, 8), %rax
	adc	%rdx, %rax
	sbb	r2, r2
	sub	$4, n
	jc	L(reduce_three)

	mov	B2modb, B2mb
	sub	b, B2mb

	ALIGN(16)
L(top):	and	B2modb, r2
	lea	(B2mb, r0), t0
	mul	B2modb
	add	r0, r2
	mov	(ap, n, 8), r0
	cmovc	t0, r2
	add	%rax, r0
	mov	r2, %rax
	adc	%rdx, %rax
	sbb	r2, r2
	sub	$1, n
	jnc	L(top)

L(reduce_three):
	C Eliminate r2
	and	b, r2
	sub	r2, %rax

L(reduce_two):
	mov	8(pre), R32(%rcx)
	test	R32(%rcx), R32(%rcx)
	jz	L(normalized)

	C Unnormalized, use B1modb to reduce to size < B (b+1)
	mulq	16(pre)
	xor	t0, t0
	add	%rax, r0
	adc	%rdx, t0
	mov	t0, %rax

	C Left-shift to normalize
ifdef(`SHLD_SLOW',`
	shl	R8(%rcx), %rax
	mov	r0, t0
	neg	R32(%rcx)
	shr	R8(%rcx), t0
	or	t0, %rax
	neg	R32(%rcx)
',`
	shld	R8(%rcx), r0, %rax
')
	shl	R8(%rcx), r0
	jmp	L(udiv)

L(normalized):
	mov	%rax, t0
	sub	b, t0
	cmovnc	t0, %rax

L(udiv):
	lea	1(%rax), t0
	mulq	(pre)
	add	r0, %rax
	adc	t0, %rdx
	imul	b, %rdx
	sub	%rdx, r0
	cmp	r0, %rax
	lea	(b, r0), %rax
	cmovnc	r0, %rax
	cmp	b, %rax
	jnc	L(fix)
L(ok):	shr	R8(%rcx), %rax

	pop	%rbx
	pop	%rbp
	FUNC_EXIT()
	ret
L(fix):	sub	b, %rax
	jmp	L(ok)
EPILOGUE()

	ALIGN(16)
PROLOGUE(mpn_mod_1_1p_cps)
	FUNC_ENTRY(2)
	push	%rbp
	bsr	%rsi, %rcx
	push	%rbx
	mov	%rdi, %rbx
	push	%r12
	xor	$63, R32(%rcx)
	mov	%rsi, %r12
	mov	R32(%rcx), R32(%rbp)
	sal	R8(%rcx), %r12
IFSTD(`	mov	%r12, %rdi	')	C pass parameter
IFDOS(`	mov	%r12, %rcx	')	C pass parameter
	ASSERT(nz, `test $15, %rsp')
	CALL(	mpn_invert_limb)
	neg	%r12
	mov	%r12, %r8
	mov	%rax, (%rbx)		C store bi
	mov	%rbp, 8(%rbx)		C store cnt
	imul	%rax, %r12
	mov	%r12, 24(%rbx)		C store B2modb
	mov	R32(%rbp), R32(%rcx)
	test	R32(%rcx), R32(%rcx)
	jz	L(z)

	mov	$1, R32(%rdx)
ifdef(`SHLD_SLOW',`
	C Destroys %rax, unlike shld. Otherwise, we could do B1modb
	C before B2modb, and get rid of the move %r12, %r8 above.

	shl	R8(%rcx), %rdx
	neg	R32(%rcx)
	shr	R8(%rcx), %rax
	or	%rax, %rdx
	neg	R32(%rcx)
',`
	shld	R8(%rcx), %rax, %rdx
')
	imul	%rdx, %r8
	shr	R8(%rcx), %r8
	mov	%r8, 16(%rbx)		C store B1modb
L(z):
	pop	%r12
	pop	%rbx
	pop	%rbp
	FUNC_EXIT()
	ret
EPILOGUE()
ASM_END()
