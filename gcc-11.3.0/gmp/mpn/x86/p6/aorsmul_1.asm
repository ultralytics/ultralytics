dnl  Intel P6 mpn_addmul_1/mpn_submul_1 -- add or subtract mpn multiple.

dnl  Copyright 1999-2002, 2005 Free Software Foundation, Inc.

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


C			    cycles/limb
C P5
C P6 model 0-8,10-12		 6.44
C P6 model 9  (Banias)		 6.15
C P6 model 13 (Dothan)		 6.11
C P4 model 0  (Willamette)
C P4 model 1  (?)
C P4 model 2  (Northwood)
C P4 model 3  (Prescott)
C P4 model 4  (Nocona)
C AMD K6
C AMD K7
C AMD K8


dnl  P6 UNROLL_COUNT cycles/limb
dnl          8           6.7
dnl         16           6.35
dnl         32           6.3
dnl         64           6.3
dnl  Maximum possible with the current code is 64.

deflit(UNROLL_COUNT, 16)


ifdef(`OPERATION_addmul_1', `
	define(M4_inst,        addl)
	define(M4_function_1,  mpn_addmul_1)
	define(M4_function_1c, mpn_addmul_1c)
	define(M4_description, add it to)
	define(M4_desc_retval, carry)
',`ifdef(`OPERATION_submul_1', `
	define(M4_inst,        subl)
	define(M4_function_1,  mpn_submul_1)
	define(M4_function_1c, mpn_submul_1c)
	define(M4_description, subtract it from)
	define(M4_desc_retval, borrow)
',`m4_error(`Need OPERATION_addmul_1 or OPERATION_submul_1
')')')

MULFUNC_PROLOGUE(mpn_addmul_1 mpn_addmul_1c mpn_submul_1 mpn_submul_1c)


C mp_limb_t M4_function_1 (mp_ptr dst, mp_srcptr src, mp_size_t size,
C                            mp_limb_t mult);
C mp_limb_t M4_function_1c (mp_ptr dst, mp_srcptr src, mp_size_t size,
C                             mp_limb_t mult, mp_limb_t carry);
C
C Calculate src,size multiplied by mult and M4_description dst,size.
C Return the M4_desc_retval limb from the top of the result.
C
C This code is pretty much the same as the K6 code.  The unrolled loop is
C the same, but there's just a few scheduling tweaks in the setups and the
C simple loop.
C
C A number of variations have been tried for the unrolled loop, with one or
C two carries, and with loads scheduled earlier, but nothing faster than 6
C cycles/limb has been found.

ifdef(`PIC',`
deflit(UNROLL_THRESHOLD, 5)
',`
deflit(UNROLL_THRESHOLD, 5)
')

defframe(PARAM_CARRY,     20)
defframe(PARAM_MULTIPLIER,16)
defframe(PARAM_SIZE,      12)
defframe(PARAM_SRC,       8)
defframe(PARAM_DST,       4)

	TEXT
	ALIGN(32)

PROLOGUE(M4_function_1c)
	pushl	%ebx
deflit(`FRAME',4)
	movl	PARAM_CARRY, %ebx
	jmp	L(start_nc)
EPILOGUE()

PROLOGUE(M4_function_1)
	push	%ebx
deflit(`FRAME',4)
	xorl	%ebx, %ebx	C initial carry

L(start_nc):
	movl	PARAM_SIZE, %ecx
	pushl	%esi
deflit(`FRAME',8)

	movl	PARAM_SRC, %esi
	pushl	%edi
deflit(`FRAME',12)

	movl	PARAM_DST, %edi
	pushl	%ebp
deflit(`FRAME',16)
	cmpl	$UNROLL_THRESHOLD, %ecx

	movl	PARAM_MULTIPLIER, %ebp
	jae	L(unroll)


	C simple loop
	C this is offset 0x22, so close enough to aligned
L(simple):
	C eax	scratch
	C ebx	carry
	C ecx	counter
	C edx	scratch
	C esi	src
	C edi	dst
	C ebp	multiplier

	movl	(%esi), %eax
	addl	$4, %edi

	mull	%ebp

	addl	%ebx, %eax
	adcl	$0, %edx

	M4_inst	%eax, -4(%edi)
	movl	%edx, %ebx

	adcl	$0, %ebx
	decl	%ecx

	leal	4(%esi), %esi
	jnz	L(simple)


	popl	%ebp
	popl	%edi

	popl	%esi
	movl	%ebx, %eax

	popl	%ebx
	ret



C------------------------------------------------------------------------------
C VAR_JUMP holds the computed jump temporarily because there's not enough
C registers when doing the mul for the initial two carry limbs.
C
C The add/adc for the initial carry in %ebx is necessary only for the
C mpn_add/submul_1c entry points.  Duplicating the startup code to
C eliminate this for the plain mpn_add/submul_1 doesn't seem like a good
C idea.

dnl  overlapping with parameters already fetched
define(VAR_COUNTER,`PARAM_SIZE')
define(VAR_JUMP,   `PARAM_DST')

	C this is offset 0x43, so close enough to aligned
L(unroll):
	C eax
	C ebx	initial carry
	C ecx	size
	C edx
	C esi	src
	C edi	dst
	C ebp

	movl	%ecx, %edx
	decl	%ecx

	subl	$2, %edx
	negl	%ecx

	shrl	$UNROLL_LOG2, %edx
	andl	$UNROLL_MASK, %ecx

	movl	%edx, VAR_COUNTER
	movl	%ecx, %edx

	C 15 code bytes per limb
ifdef(`PIC',`
	call	L(pic_calc)
L(here):
',`
	shll	$4, %edx
	negl	%ecx

	leal	L(entry) (%edx,%ecx,1), %edx
')
	movl	(%esi), %eax		C src low limb

	movl	%edx, VAR_JUMP
	leal	ifelse(UNROLL_BYTES,256,128+) 4(%esi,%ecx,4), %esi

	mull	%ebp

	addl	%ebx, %eax	C initial carry (from _1c)
	adcl	$0, %edx

	movl	%edx, %ebx	C high carry
	leal	ifelse(UNROLL_BYTES,256,128) (%edi,%ecx,4), %edi

	movl	VAR_JUMP, %edx
	testl	$1, %ecx
	movl	%eax, %ecx	C low carry

	cmovnz(	%ebx, %ecx)	C high,low carry other way around
	cmovnz(	%eax, %ebx)

	jmp	*%edx


ifdef(`PIC',`
L(pic_calc):
	shll	$4, %edx
	negl	%ecx

	C See mpn/x86/README about old gas bugs
	leal	(%edx,%ecx,1), %edx
	addl	$L(entry)-L(here), %edx

	addl	(%esp), %edx

	ret_internal
')


C -----------------------------------------------------------
	ALIGN(32)
L(top):
deflit(`FRAME',16)
	C eax	scratch
	C ebx	carry hi
	C ecx	carry lo
	C edx	scratch
	C esi	src
	C edi	dst
	C ebp	multiplier
	C
	C VAR_COUNTER	loop counter
	C
	C 15 code bytes per limb

	addl	$UNROLL_BYTES, %edi

L(entry):
deflit(CHUNK_COUNT,2)
forloop(`i', 0, UNROLL_COUNT/CHUNK_COUNT-1, `
	deflit(`disp0', eval(i*4*CHUNK_COUNT ifelse(UNROLL_BYTES,256,-128)))
	deflit(`disp1', eval(disp0 + 4))

Zdisp(	movl,	disp0,(%esi), %eax)
	mull	%ebp
Zdisp(	M4_inst,%ecx, disp0,(%edi))
	adcl	%eax, %ebx
	movl	%edx, %ecx
	adcl	$0, %ecx

	movl	disp1(%esi), %eax
	mull	%ebp
	M4_inst	%ebx, disp1(%edi)
	adcl	%eax, %ecx
	movl	%edx, %ebx
	adcl	$0, %ebx
')

	decl	VAR_COUNTER
	leal	UNROLL_BYTES(%esi), %esi

	jns	L(top)


deflit(`disp0',	eval(UNROLL_BYTES ifelse(UNROLL_BYTES,256,-128)))

	M4_inst	%ecx, disp0(%edi)
	movl	%ebx, %eax

	popl	%ebp
	popl	%edi

	popl	%esi
	popl	%ebx
	adcl	$0, %eax

	ret

EPILOGUE()
