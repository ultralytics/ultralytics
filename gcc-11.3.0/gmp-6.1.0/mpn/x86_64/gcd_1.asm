dnl  AMD64 mpn_gcd_1 -- mpn by 1 gcd.

dnl  Based on the K7 gcd_1.asm, by Kevin Ryde.  Rehacked for AMD64 by Torbjorn
dnl  Granlund.

dnl  Copyright 2000-2002, 2005, 2009, 2011, 2012 Free Software Foundation, Inc.

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


C	     cycles/bit (approx)
C AMD K8,K9	 5.21                 (4.95)
C AMD K10	 5.15                 (5.00)
C AMD bd1	 5.42                 (5.14)
C AMD bobcat	 6.71                 (6.56)
C Intel P4	13.5                 (12.75)
C Intel core2	 6.20                 (6.16)
C Intel NHM	 6.49                 (6.25)
C Intel SBR	 7.75                 (7.57)
C Intel atom	 8.77                 (8.54)
C VIA nano	 6.60                 (6.20)
C Numbers measured with: speed -CD -s16-64 -t48 mpn_gcd_1

C ctz_table[n] is the number of trailing zeros on n, or MAXSHIFT if n==0.

deflit(MAXSHIFT, 7)
deflit(MASK, eval((m4_lshift(1,MAXSHIFT))-1))

DEF_OBJECT(ctz_table,64)
	.byte	MAXSHIFT
forloop(i,1,MASK,
`	.byte	m4_count_trailing_zeros(i)
')
END_OBJECT(ctz_table)

C Threshold of when to call bmod when U is one limb.  Should be about
C (time_in_cycles(bmod_1,1) + call_overhead) / (cycles/bit).
define(`BMOD_THRES_LOG2', 8)

C INPUT PARAMETERS
define(`up',    `%rdi')
define(`n',     `%rsi')
define(`v0',    `%rdx')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

IFDOS(`define(`STACK_ALLOC', 40)')
IFSTD(`define(`STACK_ALLOC', 8)')

ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_gcd_1)
	FUNC_ENTRY(3)
	mov	(up), %rax		C U low limb
	mov	$-1, R32(%rcx)
	or	v0, %rax		C x | y

L(twos):
	inc	R32(%rcx)
	shr	%rax
	jnc	L(twos)

	shr	R8(%rcx), v0
	push	%rcx			C common twos

L(divide_strip_y):
	shr	v0
	jnc	L(divide_strip_y)
	adc	v0, v0

	cmp	$1, n
	jnz	L(reduce_nby1)

C Both U and V are single limbs, reduce with bmod if u0 >> v0.
	mov	(up), %r8
	mov	%r8, %rax
	shr	$BMOD_THRES_LOG2, %r8
	cmp	%r8, v0
	ja	L(noreduce)
	push	v0
	sub	$STACK_ALLOC, %rsp	C maintain ABI required rsp alignment

L(bmod):
IFDOS(`	mov	%rdx, %r8	')
IFDOS(`	mov	%rsi, %rdx	')
IFDOS(`	mov	%rdi, %rcx	')
	ASSERT(nz, `test $15, %rsp')
	CALL(	mpn_modexact_1_odd)

L(reduced):
	add	$STACK_ALLOC, %rsp
	pop	%rdx

L(noreduce):
	LEA(	ctz_table, %rsi)
	test	%rax, %rax
	mov	%rax, %rcx
	jnz	L(mid)
	jmp	L(end)

L(reduce_nby1):
	push	v0
	sub	$STACK_ALLOC, %rsp	C maintain ABI required rsp alignment

	cmp	$BMOD_1_TO_MOD_1_THRESHOLD, n
	jl	L(bmod)
IFDOS(`	mov	%rdx, %r8	')
IFDOS(`	mov	%rsi, %rdx	')
IFDOS(`	mov	%rdi, %rcx	')
	ASSERT(nz, `test $15, %rsp')
	CALL(	mpn_mod_1)
	jmp	L(reduced)

	ALIGN(16)			C               K8    BC    P4    NHM   SBR
L(top):	cmovc	%rcx, %rax		C if x-y < 0	0
	cmovc	%rdi, %rdx		C use x,y-x	0
L(mid):	and	$MASK, R32(%rcx)	C		0
	movzbl	(%rsi,%rcx), R32(%rcx)	C		1
	jz	L(shift_alot)		C		1
	shr	R8(%rcx), %rax		C		3
	mov	%rax, %rdi		C		4
	mov	%rdx, %rcx		C		3
	sub	%rax, %rcx		C		4
	sub	%rdx, %rax		C		4
	jnz	L(top)			C		5

L(end):	pop	%rcx
	mov	%rdx, %rax
	shl	R8(%rcx), %rax
	FUNC_EXIT()
	ret

L(shift_alot):
	shr	$MAXSHIFT, %rax
	mov	%rax, %rcx
	jmp	L(mid)
EPILOGUE()
