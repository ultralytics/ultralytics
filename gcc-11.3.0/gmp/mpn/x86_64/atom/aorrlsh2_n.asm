dnl  AMD64 mpn_addlsh2_n -- rp[] = up[] + (vp[] << 2)
dnl  AMD64 mpn_rsblsh2_n -- rp[] = (vp[] << 2) - up[]
dnl  Optimised for Intel Atom.

dnl  Contributed to the GNU project by Torbjorn Granlund.

dnl  Copyright 2011, 2012 Free Software Foundation, Inc.

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
C Intel core2	 ?
C Intel NHM	 ?
C Intel SBR	 ?
C Intel atom	 5.75
C VIA nano	 ?

C INPUT PARAMETERS
define(`rp',       `%rdi')
define(`up',       `%rsi')
define(`vp',       `%rdx')
define(`n',        `%rcx')

define(`LSH', 2)
define(`RSH', 62)
define(M, eval(m4_lshift(1,LSH)))

ifdef(`OPERATION_addlsh2_n', `
  define(ADDSUB,	add)
  define(ADCSBB,	adc)
  define(func_n,	mpn_addlsh2_n)
  define(func_nc,	mpn_addlsh2_nc)')
ifdef(`OPERATION_rsblsh2_n', `
  define(ADDSUB,	sub)
  define(ADCSBB,	sbb)
  define(func_n,	mpn_rsblsh2_n)
  define(func_nc,	mpn_rsblsh2_nc)')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

MULFUNC_PROLOGUE(mpn_addlsh2_n mpn_rsblsh2_n)

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(func_n)
	FUNC_ENTRY(4)
	push	%rbx
	push	%rbp

	mov	R32(n), R32(%rax)
	and	$3, R32(%rax)
	jz	L(b0)			C we rely on rax = 0 at target
	cmp	$2, R32(%rax)
	mov	$0, R32(%rax)
	jz	L(b2)
	jg	L(b3)

L(b1):	mov	(vp), %r9
	lea	(%rax,%r9,M), %rbp
	shr	$RSH, %r9
	sub	$1, n
	lea	-8(up), up
	lea	-8(rp), rp
	jz	L(cj1)
	mov	8(vp), %r10
	lea	(%r9,%r10,M), %r9
	shr	$RSH, %r10
	mov	16(vp), %r11
	lea	24(vp), vp
	mov	(vp), %r8
	lea	(%r10,%r11,M), %r10
	shr	$RSH, %r11
	add	R32(%rax), R32(%rax)
	jmp	L(L1)

L(b2):	lea	-32(rp), rp
	mov	(vp), %r8
	lea	-32(up), up
	lea	(%rax,%r8,M), %rbx
	shr	$RSH, %r8
	mov	8(vp), %r9
	sub	$2, n
	jle	L(end)
	jmp	L(top)

L(b3):	lea	-24(up), up
	mov	(vp), %r11
	lea	-24(rp), rp
	mov	8(vp), %r8
	lea	(%rax,%r11,M), %r10
	shr	$RSH, %r11
	lea	8(vp), vp
	lea	(%r11,%r8,M), %rbx
	add	$1, n
	jmp	L(L3)

L(b0):	lea	-16(up), up
	mov	(vp), %r10
	lea	(%rax,%r10,M), %r9
	shr	$RSH, %r10
	mov	8(vp), %r11
	lea	-16(rp), rp
	mov	16(vp), %r8
	lea	(%r10,%r11,M), %r10
	shr	$RSH, %r11
	add	R32(%rax), R32(%rax)
	lea	16(vp), vp
	jmp	L(L0)

	ALIGN(16)
L(top):	lea	(%r8,%r9,M), %rbp
	shr	$RSH, %r9
	lea	32(up), up
	mov	16(vp), %r10
	lea	(%r9,%r10,M), %r9
	shr	$RSH, %r10
	mov	24(vp), %r11
	lea	32(rp), rp
	lea	32(vp), vp
	mov	(vp), %r8
	lea	(%r10,%r11,M), %r10
	shr	$RSH, %r11
	add	R32(%rax), R32(%rax)
	ADCSBB	(up), %rbx
	mov	%rbx, (rp)
L(L1):	ADCSBB	8(up), %rbp
	mov	%rbp, 8(rp)
L(L0):	ADCSBB	16(up), %r9
	lea	(%r11,%r8,M), %rbx
	mov	%r9, 16(rp)
L(L3):	ADCSBB	24(up), %r10
	sbb	R32(%rax), R32(%rax)
L(L2):	shr	$RSH, %r8
	mov	8(vp), %r9
	mov	%r10, 24(rp)
	sub	$4, n
	jg	L(top)

L(end):	lea	(%r8,%r9,M), %rbp
	shr	$RSH, %r9
	lea	32(up), up
	lea	32(rp), rp
	add	R32(%rax), R32(%rax)
	ADCSBB	(up), %rbx
	mov	%rbx, (rp)
L(cj1):	ADCSBB	8(up), %rbp
	mov	%rbp, 8(rp)

ifdef(`OPERATION_addlsh2_n',`
	mov	R32(n), R32(%rax)	C zero rax
	adc	%r9, %rax')
ifdef(`OPERATION_rsblsh2_n',`
	sbb	n, %r9			C subtract 0
	mov	%r9, %rax')

	pop	%rbp
	pop	%rbx
	FUNC_EXIT()
	ret
EPILOGUE()
