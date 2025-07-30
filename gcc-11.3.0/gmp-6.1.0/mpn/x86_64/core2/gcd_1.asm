dnl  AMD64 mpn_gcd_1 optimised for Intel C2, NHM, SBR and AMD K10, BD.

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
C AMD K8,K9	 8.50
C AMD K10	 4.30
C AMD bd1	 5.00
C AMD bobcat	10.0
C Intel P4	18.6
C Intel core2	 3.83
C Intel NHM	 5.17
C Intel SBR	 4.69
C Intel atom	17.0
C VIA nano	 5.44
C Numbers measured with: speed -CD -s16-64 -t48 mpn_gcd_1

C TODO
C  * Optimise inner-loop for specific CPUs.
C  * Use DIV for 1-by-1 reductions, at least for some CPUs.

C Threshold of when to call bmod when U is one limb.  Should be about
C (time_in_cycles(bmod_1,1) + call_overhead) / (cycles/bit).
define(`BMOD_THRES_LOG2', 6)

C INPUT PARAMETERS
define(`up',    `%rdi')
define(`n',     `%rsi')
define(`v0',    `%rdx')

ABI_SUPPORT(DOS64)
ABI_SUPPORT(STD64)

IFDOS(`define(`STACK_ALLOC', 40)')
IFSTD(`define(`STACK_ALLOC', 8)')

C Undo some configure cleverness.
C The problem is that C only defines the '1c' variant, and that configure
C therefore considers modexact_1c to be the base function.  It then adds a
C special fat rule for mpn_modexact_1_odd, messing up things when a cpudep
C gcd_1 exists without a corresponding cpudep mode1o.
ifdef(`WANT_FAT_BINARY', `
  define(`mpn_modexact_1_odd', `MPN_PREFIX`modexact_1_odd_x86_64'')')


ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_gcd_1)
	FUNC_ENTRY(3)
	mov	(up), %rax	C U low limb
	or	v0, %rax
	bsf	%rax, %rax	C min(ctz(u0),ctz(v0))

	bsf	v0, %rcx
	shr	R8(%rcx), v0

	push	%rax		C preserve common twos over call
	push	v0		C preserve v0 argument over call
	sub	$STACK_ALLOC, %rsp	C maintain ABI required rsp alignment

	cmp	$1, n
	jnz	L(reduce_nby1)

C Both U and V are single limbs, reduce with bmod if u0 >> v0.
	mov	(up), %r8
	mov	%r8, %rax
	shr	$BMOD_THRES_LOG2, %r8
	cmp	%r8, v0
	ja	L(reduced)
	jmp	L(bmod)

L(reduce_nby1):
	cmp	$BMOD_1_TO_MOD_1_THRESHOLD, n
	jl	L(bmod)
IFDOS(`	mov	%rdx, %r8	')
IFDOS(`	mov	%rsi, %rdx	')
IFDOS(`	mov	%rdi, %rcx	')
	ASSERT(nz, `test $15, %rsp')
	CALL(	mpn_mod_1)
	jmp	L(reduced)
L(bmod):
IFDOS(`	mov	%rdx, %r8	')
IFDOS(`	mov	%rsi, %rdx	')
IFDOS(`	mov	%rdi, %rcx	')
	ASSERT(nz, `test $15, %rsp')
	CALL(	mpn_modexact_1_odd)
L(reduced):

	add	$STACK_ALLOC, %rsp
	pop	%rdx

	bsf	%rax, %rcx
C	test	%rax, %rax	C FIXME: does this lower latency?
	jnz	L(mid)
	jmp	L(end)

	ALIGN(16)		C               K10   BD    C2    NHM   SBR
L(top):	cmovc	%r10, %rax	C if x-y < 0    0,3   0,3   0,6   0,5   0,5
	cmovc	%r9, %rdx	C use x,y-x     0,3   0,3   2,8   1,7   1,7
L(mid):	shr	R8(%rcx), %rax	C               1,7   1,6   2,8   2,8   2,8
	mov	%rdx, %r10	C               1     1     4     3     3
	sub	%rax, %r10	C               2     2     5     4     4
	bsf	%r10, %rcx	C               3     3     6     5     5
	mov	%rax, %r9	C               2     2     3     3     4
	sub	%rdx, %rax	C               2     2     4     3     4
	jnz	L(top)		C

L(end):	pop	%rcx
	mov	%rdx, %rax
	shl	R8(%rcx), %rax
	FUNC_EXIT()
	ret
EPILOGUE()
