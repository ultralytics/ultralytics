dnl  x86 mpn_gcd_1 optimised for AMD K7.

dnl  Contributed to the GNU project by by Kevin Ryde.  Rehacked by Torbjorn
dnl  Granlund.

dnl  Copyright 2000-2002, 2005, 2009, 2011, 2012, 2014, 2015 Free Software
dnl  Foundation, Inc.

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
C AMD K7	 5.31
C AMD K8,K9	 5.33
C AMD K10	 5.30
C AMD bd1	 ?
C AMD bobcat	 7.02
C Intel P4-2	10.1
C Intel P4-3/4	10.0
C Intel P6/13	 5.88
C Intel core2	 6.26
C Intel NHM	 6.83
C Intel SBR	 8.50
C Intel atom	 8.90
C VIA nano	 ?
C Numbers measured with: speed -CD -s16-32 -t16 mpn_gcd_1

C TODO
C  * Tune overhead, this takes 2-3 cycles more than old code when v0 is tiny.
C  * Stream things better through registers, avoiding some copying.
C  * For ELF, avoid putting GOT base in both ebx and esi.  Needs special
C    LEA/LEAL or else discrete code here.

C ctz_table[n] is the number of trailing zeros on n, or MAXSHIFT if n==0.

deflit(MAXSHIFT, 6)
deflit(MASK, eval((m4_lshift(1,MAXSHIFT))-1))

DEF_OBJECT(ctz_table,64)
	.byte	MAXSHIFT
forloop(i,1,MASK,
`	.byte	m4_count_trailing_zeros(i)
')
END_OBJECT(ctz_table)

C Threshold of when to call bmod when U is one limb.  Should be about
C (time_in_cycles(bmod_1,1) + call_overhead) / (cycles/bit).
define(`DIV_THRES_LOG2', 7)


define(`up',    `%edi')
define(`n',     `%esi')
define(`v0',    `%edx')


ASM_START()
	TEXT
	ALIGN(16)
PROLOGUE(mpn_gcd_1)
	push	%edi
	push	%esi

	mov	12(%esp), up
	mov	16(%esp), n
	mov	20(%esp), v0

	mov	(up), %eax		C U low limb
	or	v0, %eax		C x | y
	mov	$-1, %ecx

L(twos):
	inc	%ecx
	shr	%eax
	jnc	L(twos)

	shr	%cl, v0
	mov	%ecx, %eax		C common twos

L(divide_strip_y):
	shr	v0
	jnc	L(divide_strip_y)
	adc	v0, v0

	push	%eax
	push	v0

	cmp	$1, n
	jnz	L(reduce_nby1)

C Both U and V are single limbs, reduce with bmod if u0 >> v0.
	mov	(up), %ecx
	mov	%ecx, %eax
	shr	$DIV_THRES_LOG2, %ecx
	cmp	%ecx, v0
	ja	L(reduced)

	mov	v0, %esi
	xor	%edx, %edx
	div	%esi
	mov	%edx, %eax
	jmp	L(reduced)

L(reduce_nby1):
ifdef(`PIC_WITH_EBX',`dnl
	push	%ebx
	add	$-4, %esp
	call	L(movl_eip_ebx)
	add	$_GLOBAL_OFFSET_TABLE_, %ebx
')
	push	v0			C param 3
	push	n			C param 2
	push	up			C param 1
	cmp	$BMOD_1_TO_MOD_1_THRESHOLD, n
	jl	L(bmod)
	CALL(	mpn_mod_1)
	jmp	L(called)
L(bmod):
	CALL(	mpn_modexact_1_odd)

L(called):
ifdef(`PIC_WITH_EBX',`dnl
	add	$16, %esp	C deallocate params
	pop	%ebx
',`
	add	$12, %esp		C deallocate params
')
L(reduced):
	pop	%edx

	LEAL(	ctz_table, %esi)
	test	%eax, %eax
	mov	%eax, %ecx
	jnz	L(mid)
	jmp	L(end)

	ALIGN(16)			C               K8    BC    P4    NHM   SBR
L(top):	cmovc(	%ecx, %eax)		C if x-y < 0	0
	cmovc(	%edi, %edx)		C use x,y-x	0
L(mid):	and	$MASK, %ecx		C		0
	movzbl	(%esi,%ecx), %ecx	C		1
	jz	L(shift_alot)		C		1
	shr	%cl, %eax		C		3
	mov	%eax, %edi		C		4
	mov	%edx, %ecx		C		3
	sub	%eax, %ecx		C		4
	sub	%edx, %eax		C		4
	jnz	L(top)			C		5

L(end):	pop	%ecx
	mov	%edx, %eax
	shl	%cl, %eax
	pop	%esi
	pop	%edi
	ret

L(shift_alot):
	shr	$MAXSHIFT, %eax
	mov	%eax, %ecx
	jmp	L(mid)

ifdef(`PIC_WITH_EBX',`dnl
L(movl_eip_ebx):
	mov	(%esp), %ebx
	ret
')
EPILOGUE()
ASM_END()
