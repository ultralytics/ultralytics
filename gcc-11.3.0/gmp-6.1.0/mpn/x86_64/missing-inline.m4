dnl  AMD64 MULX/ADX simulation support, inline version.

dnl  Contributed to the GNU project by Torbjörn Granlund.

dnl  Copyright 2013 Free Software Foundation, Inc.

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


define(`adox',`
	push	$2
	push	%rcx
	push	%rbx
	push	%rax
	mov	$1, %rcx
	pushfq
	pushfq
C copy 0(%rsp):11 to 0(%rsp):0
	mov	(%rsp), %rbx
	shr	%rbx
	bt	$`'10, %rbx
	adc	%rbx, %rbx
	mov	%rbx, (%rsp)
C put manipulated flags into eflags, execute a plain adc
	popfq
	adc	%rcx, 32(%rsp)
C copy CF to 0(%rsp):11
	mov	(%rsp), %rbx
	sbb	R32(%rax), R32(%rax)
	and	$`'0x800, R32(%rax)
	and	$`'0xfffffffffffff7ff, %rbx
	or	%rax, %rbx
	mov	%rbx, (%rsp)
C put manipulated flags into eflags
	popfq
	pop	%rax
	pop	%rbx
	pop	%rcx
	pop	$2
')

define(`adcx',`
	push	$2
	push	%rcx
	push	%rbx
	push	%rax
	mov	$1, %rcx
	pushfq
	adc	%rcx, 32(%rsp)
	mov	(%rsp), %rbx
	sbb	R32(%rax), R32(%rax)
	and	$`'0xfffffffffffffffe, %rbx
	sub	%rax, %rbx
	mov	%rbx, (%rsp)
	popfq
	pop	%rax
	pop	%rbx
	pop	%rcx
	pop	$2
')

define(`mulx',`
	lea	-16(%rsp), %rsp
	push	%rax
	push	%rdx
	pushfq			C preserve all flags
	mov	$1, %rax
	mul	%rdx
	mov	%rax, 24(%rsp)
	mov	%rdx, 32(%rsp)
	popfq			C restore eflags
	pop	%rdx
	pop	%rax
	pop	$2
	pop	$3
')
