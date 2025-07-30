dnl  AMD64 mpn_addlsh_n and mpn_rsblsh_n.  R = V2^k +- U.

dnl  Copyright 2006, 2010-2012 Free Software Foundation, Inc.

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
C AMD K8,K9	 3.1	< 3.85 for lshift + add_n
C AMD K10	 3.1	< 3.85 for lshift + add_n
C Intel P4	14.6	> 7.33 for lshift + add_n
C Intel core2	 3.87	> 3.27 for lshift + add_n
C Intel NHM	 4	> 3.75 for lshift + add_n
C Intel SBR	(5.8)	> 3.46 for lshift + add_n
C Intel atom	(7.75)	< 8.75 for lshift + add_n
C VIA nano	 4.7	< 6.25 for lshift + add_n

C This was written quickly and not optimized at all.  Surely one could get
C closer to 3 c/l or perhaps even under 3 c/l.  Ideas:
C   1) Use indexing to save the 3 LEA
C   2) Write reasonable feed-in code
C   3) Be more clever about register usage
C   4) Unroll more, handling CL negation, carry save/restore cost much now
C   5) Reschedule

C INPUT PARAMETERS
define(`rp',	`%rdi')
define(`up',	`%rsi')
define(`vp',	`%rdx')
define(`n',	`%rcx')
define(`cnt',	`%r8')

ifdef(`OPERATION_addlsh_n',`
  define(ADCSBB,       `adc')
  define(func, mpn_addlsh_n)
')
ifdef(`OPERATION_rsblsh_n',`
  define(ADCSBB,       `sbb')
  define(func, mpn_rsblsh_n)
')

MULFUNC_PROLOGUE(mpn_addlsh_n mpn_rsblsh_n)

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(func)
	FUNC_ENTRY(4)
IFDOS(`	mov	56(%rsp), %r8d	')
	push	%r12
	push	%r13
	push	%r14
	push	%rbp
	push	%rbx

	mov	n, %rax
	xor	R32(%rbx), R32(%rbx)	C clear carry save register
	mov	R32(%r8), R32(%rcx)	C shift count
	xor	R32(%rbp), R32(%rbp)	C limb carry

	mov	R32(%rax), R32(%r11)
	and	$3, R32(%r11)
	je	L(4)
	sub	$1, R32(%r11)

L(012):	mov	(vp), %r8
	mov	%r8, %r12
	shl	R8(%rcx), %r8
	or	%rbp, %r8
	neg	R8(%rcx)
	mov	%r12, %rbp
	shr	R8(%rcx), %rbp
	neg	R8(%rcx)
	add	R32(%rbx), R32(%rbx)
	ADCSBB	(up), %r8
	mov	%r8, (rp)
	sbb	R32(%rbx), R32(%rbx)
	lea	8(up), up
	lea	8(vp), vp
	lea	8(rp), rp
	sub	$1, R32(%r11)
	jnc	L(012)

L(4):	sub	$4, %rax
	jc	L(end)

	ALIGN(16)
L(top):	mov	(vp), %r8
	mov	%r8, %r12
	mov	8(vp), %r9
	mov	%r9, %r13
	mov	16(vp), %r10
	mov	%r10, %r14
	mov	24(vp), %r11

	shl	R8(%rcx), %r8
	shl	R8(%rcx), %r9
	shl	R8(%rcx), %r10
	or	%rbp, %r8
	mov	%r11, %rbp
	shl	R8(%rcx), %r11

	neg	R8(%rcx)

	shr	R8(%rcx), %r12
	shr	R8(%rcx), %r13
	shr	R8(%rcx), %r14
	shr	R8(%rcx), %rbp		C used next iteration

	or	%r12, %r9
	or	%r13, %r10
	or	%r14, %r11

	neg	R8(%rcx)

	add	R32(%rbx), R32(%rbx)	C restore carry flag

	ADCSBB	(up), %r8
	ADCSBB	8(up), %r9
	ADCSBB	16(up), %r10
	ADCSBB	24(up), %r11

	mov	%r8, (rp)
	mov	%r9, 8(rp)
	mov	%r10, 16(rp)
	mov	%r11, 24(rp)

	sbb	R32(%rbx), R32(%rbx)	C save carry flag

	lea	32(up), up
	lea	32(vp), vp
	lea	32(rp), rp

	sub	$4, %rax
	jnc	L(top)

L(end):	add	R32(%rbx), R32(%rbx)
	ADCSBB	$0, %rbp
	mov	%rbp, %rax
	pop	%rbx
	pop	%rbp
	pop	%r14
	pop	%r13
	pop	%r12
	FUNC_EXIT()
	ret
EPILOGUE()
