dnl  Intel P6 mpn_sqr_basecase -- square an mpn number.

dnl  Copyright 1999, 2000, 2002 Free Software Foundation, Inc.

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


C P6: approx 4.0 cycles per cross product, or 7.75 cycles per triangular
C     product (measured on the speed difference between 20 and 40 limbs,
C     which is the Karatsuba recursing range).


dnl  These are the same as in mpn/x86/k6/sqr_basecase.asm, see that file for
dnl  a description.  The only difference here is that UNROLL_COUNT can go up
dnl  to 64 (not 63) making SQR_TOOM2_THRESHOLD_MAX 67.

deflit(SQR_TOOM2_THRESHOLD_MAX, 67)

ifdef(`SQR_TOOM2_THRESHOLD_OVERRIDE',
`define(`SQR_TOOM2_THRESHOLD',SQR_TOOM2_THRESHOLD_OVERRIDE)')

m4_config_gmp_mparam(`SQR_TOOM2_THRESHOLD')
deflit(UNROLL_COUNT, eval(SQR_TOOM2_THRESHOLD-3))


C void mpn_sqr_basecase (mp_ptr dst, mp_srcptr src, mp_size_t size);
C
C The algorithm is basically the same as mpn/generic/sqr_basecase.c, but a
C lot of function call overheads are avoided, especially when the given size
C is small.
C
C The code size might look a bit excessive, but not all of it is executed so
C it won't all get into the code cache.  The 1x1, 2x2 and 3x3 special cases
C clearly apply only to those sizes; mid sizes like 10x10 only need part of
C the unrolled addmul; and big sizes like 40x40 that do use the full
C unrolling will least be making good use of it, because 40x40 will take
C something like 7000 cycles.

defframe(PARAM_SIZE,12)
defframe(PARAM_SRC, 8)
defframe(PARAM_DST, 4)

	TEXT
	ALIGN(32)
PROLOGUE(mpn_sqr_basecase)
deflit(`FRAME',0)

	movl	PARAM_SIZE, %edx

	movl	PARAM_SRC, %eax

	cmpl	$2, %edx
	movl	PARAM_DST, %ecx
	je	L(two_limbs)

	movl	(%eax), %eax
	ja	L(three_or_more)


C -----------------------------------------------------------------------------
C one limb only
	C eax	src limb
	C ebx
	C ecx	dst
	C edx

	mull	%eax

	movl	%eax, (%ecx)
	movl	%edx, 4(%ecx)

	ret


C -----------------------------------------------------------------------------
L(two_limbs):
	C eax	src
	C ebx
	C ecx	dst
	C edx

defframe(SAVE_ESI, -4)
defframe(SAVE_EBX, -8)
defframe(SAVE_EDI, -12)
defframe(SAVE_EBP, -16)
deflit(`STACK_SPACE',16)

	subl	$STACK_SPACE, %esp
deflit(`FRAME',STACK_SPACE)

	movl	%esi, SAVE_ESI
	movl	%eax, %esi
	movl	(%eax), %eax

	mull	%eax		C src[0]^2

	movl	%eax, (%ecx)	C dst[0]
	movl	4(%esi), %eax

	movl	%ebx, SAVE_EBX
	movl	%edx, %ebx	C dst[1]

	mull	%eax		C src[1]^2

	movl	%edi, SAVE_EDI
	movl	%eax, %edi	C dst[2]
	movl	(%esi), %eax

	movl	%ebp, SAVE_EBP
	movl	%edx, %ebp	C dst[3]

	mull	4(%esi)		C src[0]*src[1]

	addl	%eax, %ebx
	movl	SAVE_ESI, %esi

	adcl	%edx, %edi

	adcl	$0, %ebp
	addl	%ebx, %eax
	movl	SAVE_EBX, %ebx

	adcl	%edi, %edx
	movl	SAVE_EDI, %edi

	adcl	$0, %ebp

	movl	%eax, 4(%ecx)

	movl	%ebp, 12(%ecx)
	movl	SAVE_EBP, %ebp

	movl	%edx, 8(%ecx)
	addl	$FRAME, %esp

	ret


C -----------------------------------------------------------------------------
L(three_or_more):
	C eax	src low limb
	C ebx
	C ecx	dst
	C edx	size
deflit(`FRAME',0)

	pushl	%esi	defframe_pushl(`SAVE_ESI')
	cmpl	$4, %edx

	movl	PARAM_SRC, %esi
	jae	L(four_or_more)


C -----------------------------------------------------------------------------
C three limbs

	C eax	src low limb
	C ebx
	C ecx	dst
	C edx
	C esi	src
	C edi
	C ebp

	pushl	%ebp	defframe_pushl(`SAVE_EBP')
	pushl	%edi	defframe_pushl(`SAVE_EDI')

	mull	%eax		C src[0] ^ 2

	movl	%eax, (%ecx)
	movl	%edx, 4(%ecx)

	movl	4(%esi), %eax
	xorl	%ebp, %ebp

	mull	%eax		C src[1] ^ 2

	movl	%eax, 8(%ecx)
	movl	%edx, 12(%ecx)
	movl	8(%esi), %eax

	pushl	%ebx	defframe_pushl(`SAVE_EBX')

	mull	%eax		C src[2] ^ 2

	movl	%eax, 16(%ecx)
	movl	%edx, 20(%ecx)

	movl	(%esi), %eax

	mull	4(%esi)		C src[0] * src[1]

	movl	%eax, %ebx
	movl	%edx, %edi

	movl	(%esi), %eax

	mull	8(%esi)		C src[0] * src[2]

	addl	%eax, %edi
	movl	%edx, %ebp

	adcl	$0, %ebp
	movl	4(%esi), %eax

	mull	8(%esi)		C src[1] * src[2]

	xorl	%esi, %esi
	addl	%eax, %ebp

	C eax
	C ebx	dst[1]
	C ecx	dst
	C edx	dst[4]
	C esi	zero, will be dst[5]
	C edi	dst[2]
	C ebp	dst[3]

	adcl	$0, %edx
	addl	%ebx, %ebx

	adcl	%edi, %edi

	adcl	%ebp, %ebp

	adcl	%edx, %edx
	movl	4(%ecx), %eax

	adcl	$0, %esi
	addl	%ebx, %eax

	movl	%eax, 4(%ecx)
	movl	8(%ecx), %eax

	adcl	%edi, %eax
	movl	12(%ecx), %ebx

	adcl	%ebp, %ebx
	movl	16(%ecx), %edi

	movl	%eax, 8(%ecx)
	movl	SAVE_EBP, %ebp

	movl	%ebx, 12(%ecx)
	movl	SAVE_EBX, %ebx

	adcl	%edx, %edi
	movl	20(%ecx), %eax

	movl	%edi, 16(%ecx)
	movl	SAVE_EDI, %edi

	adcl	%esi, %eax	C no carry out of this
	movl	SAVE_ESI, %esi

	movl	%eax, 20(%ecx)
	addl	$FRAME, %esp

	ret



C -----------------------------------------------------------------------------
defframe(VAR_COUNTER,-20)
defframe(VAR_JMP,    -24)
deflit(`STACK_SPACE',24)

L(four_or_more):
	C eax	src low limb
	C ebx
	C ecx
	C edx	size
	C esi	src
	C edi
	C ebp
deflit(`FRAME',4)  dnl  %esi already pushed

C First multiply src[0]*src[1..size-1] and store at dst[1..size].

	subl	$STACK_SPACE-FRAME, %esp
deflit(`FRAME',STACK_SPACE)
	movl	$1, %ecx

	movl	%edi, SAVE_EDI
	movl	PARAM_DST, %edi

	movl	%ebx, SAVE_EBX
	subl	%edx, %ecx		C -(size-1)

	movl	%ebp, SAVE_EBP
	movl	$0, %ebx		C initial carry

	leal	(%esi,%edx,4), %esi	C &src[size]
	movl	%eax, %ebp		C multiplier

	leal	-4(%edi,%edx,4), %edi	C &dst[size-1]


C This loop runs at just over 6 c/l.

L(mul_1):
	C eax	scratch
	C ebx	carry
	C ecx	counter, limbs, negative, -(size-1) to -1
	C edx	scratch
	C esi	&src[size]
	C edi	&dst[size-1]
	C ebp	multiplier

	movl	%ebp, %eax

	mull	(%esi,%ecx,4)

	addl	%ebx, %eax
	movl	$0, %ebx

	adcl	%edx, %ebx
	movl	%eax, 4(%edi,%ecx,4)

	incl	%ecx
	jnz	L(mul_1)


	movl	%ebx, 4(%edi)


C Addmul src[n]*src[n+1..size-1] at dst[2*n-1...], for each n=1..size-2.
C
C The last two addmuls, which are the bottom right corner of the product
C triangle, are left to the end.  These are src[size-3]*src[size-2,size-1]
C and src[size-2]*src[size-1].  If size is 4 then it's only these corner
C cases that need to be done.
C
C The unrolled code is the same as mpn_addmul_1(), see that routine for some
C comments.
C
C VAR_COUNTER is the outer loop, running from -(size-4) to -1, inclusive.
C
C VAR_JMP is the computed jump into the unrolled code, stepped by one code
C chunk each outer loop.

dnl  This is also hard-coded in the address calculation below.
deflit(CODE_BYTES_PER_LIMB, 15)

dnl  With &src[size] and &dst[size-1] pointers, the displacements in the
dnl  unrolled code fit in a byte for UNROLL_COUNT values up to 32, but above
dnl  that an offset must be added to them.
deflit(OFFSET,
ifelse(eval(UNROLL_COUNT>32),1,
eval((UNROLL_COUNT-32)*4),
0))

	C eax
	C ebx	carry
	C ecx
	C edx
	C esi	&src[size]
	C edi	&dst[size-1]
	C ebp

	movl	PARAM_SIZE, %ecx

	subl	$4, %ecx
	jz	L(corner)

	movl	%ecx, %edx
	negl	%ecx

	shll	$4, %ecx
ifelse(OFFSET,0,,`subl	$OFFSET, %esi')

ifdef(`PIC',`
	call	L(pic_calc)
L(here):
',`
	leal	L(unroll_inner_end)-eval(2*CODE_BYTES_PER_LIMB)(%ecx,%edx), %ecx
')
	negl	%edx

ifelse(OFFSET,0,,`subl	$OFFSET, %edi')

	C The calculated jump mustn't be before the start of the available
	C code.  This is the limit that UNROLL_COUNT puts on the src operand
	C size, but checked here using the jump address directly.

	ASSERT(ae,
	`movl_text_address( L(unroll_inner_start), %eax)
	cmpl	%eax, %ecx')


C -----------------------------------------------------------------------------
	ALIGN(16)
L(unroll_outer_top):
	C eax
	C ebx	high limb to store
	C ecx	VAR_JMP
	C edx	VAR_COUNTER, limbs, negative
	C esi	&src[size], constant
	C edi	dst ptr, second highest limb of last addmul
	C ebp

	movl	-12+OFFSET(%esi,%edx,4), %ebp	C multiplier
	movl	%edx, VAR_COUNTER

	movl	-8+OFFSET(%esi,%edx,4), %eax	C first limb of multiplicand

	mull	%ebp

define(cmovX,`ifelse(eval(UNROLL_COUNT%2),1,`cmovz($@)',`cmovnz($@)')')

	testb	$1, %cl

	movl	%edx, %ebx	C high carry
	leal	4(%edi), %edi

	movl	%ecx, %edx	C jump

	movl	%eax, %ecx	C low carry
	leal	CODE_BYTES_PER_LIMB(%edx), %edx

	cmovX(	%ebx, %ecx)	C high carry reverse
	cmovX(	%eax, %ebx)	C low carry reverse
	movl	%edx, VAR_JMP
	jmp	*%edx


	C Must be on an even address here so the low bit of the jump address
	C will indicate which way around ecx/ebx should start.

	ALIGN(2)

L(unroll_inner_start):
	C eax	scratch
	C ebx	carry high
	C ecx	carry low
	C edx	scratch
	C esi	src pointer
	C edi	dst pointer
	C ebp	multiplier
	C
	C 15 code bytes each limb
	C ecx/ebx reversed on each chunk

forloop(`i', UNROLL_COUNT, 1, `
	deflit(`disp_src', eval(-i*4 + OFFSET))
	deflit(`disp_dst', eval(disp_src))

	m4_assert(`disp_src>=-128 && disp_src<128')
	m4_assert(`disp_dst>=-128 && disp_dst<128')

ifelse(eval(i%2),0,`
Zdisp(	movl,	disp_src,(%esi), %eax)
	mull	%ebp
Zdisp(	addl,	%ebx, disp_dst,(%edi))
	adcl	%eax, %ecx
	movl	%edx, %ebx
	adcl	$0, %ebx
',`
	dnl  this one comes out last
Zdisp(	movl,	disp_src,(%esi), %eax)
	mull	%ebp
Zdisp(	addl,	%ecx, disp_dst,(%edi))
	adcl	%eax, %ebx
	movl	%edx, %ecx
	adcl	$0, %ecx
')
')
L(unroll_inner_end):

	addl	%ebx, m4_empty_if_zero(OFFSET)(%edi)

	movl	VAR_COUNTER, %edx
	adcl	$0, %ecx

	movl	%ecx, m4_empty_if_zero(OFFSET+4)(%edi)
	movl	VAR_JMP, %ecx

	incl	%edx
	jnz	L(unroll_outer_top)


ifelse(OFFSET,0,,`
	addl	$OFFSET, %esi
	addl	$OFFSET, %edi
')


C -----------------------------------------------------------------------------
	ALIGN(16)
L(corner):
	C eax
	C ebx
	C ecx
	C edx
	C esi	&src[size]
	C edi	&dst[2*size-5]
	C ebp

	movl	-12(%esi), %eax

	mull	-8(%esi)

	addl	%eax, (%edi)
	movl	-12(%esi), %eax
	movl	$0, %ebx

	adcl	%edx, %ebx

	mull	-4(%esi)

	addl	%eax, %ebx
	movl	-8(%esi), %eax

	adcl	$0, %edx

	addl	%ebx, 4(%edi)
	movl	$0, %ebx

	adcl	%edx, %ebx

	mull	-4(%esi)

	movl	PARAM_SIZE, %ecx
	addl	%ebx, %eax

	adcl	$0, %edx

	movl	%eax, 8(%edi)

	movl	%edx, 12(%edi)
	movl	PARAM_DST, %edi


C Left shift of dst[1..2*size-2], the bit shifted out becomes dst[2*size-1].

	subl	$1, %ecx		C size-1
	xorl	%eax, %eax		C ready for final adcl, and clear carry

	movl	%ecx, %edx
	movl	PARAM_SRC, %esi


L(lshift):
	C eax
	C ebx
	C ecx	counter, size-1 to 1
	C edx	size-1 (for later use)
	C esi	src (for later use)
	C edi	dst, incrementing
	C ebp

	rcll	4(%edi)
	rcll	8(%edi)

	leal	8(%edi), %edi
	decl	%ecx
	jnz	L(lshift)


	adcl	%eax, %eax

	movl	%eax, 4(%edi)		C dst most significant limb
	movl	(%esi), %eax		C src[0]

	leal	4(%esi,%edx,4), %esi	C &src[size]
	subl	%edx, %ecx		C -(size-1)


C Now add in the squares on the diagonal, src[0]^2, src[1]^2, ...,
C src[size-1]^2.  dst[0] hasn't yet been set at all yet, and just gets the
C low limb of src[0]^2.


	mull	%eax

	movl	%eax, (%edi,%ecx,8)	C dst[0]


L(diag):
	C eax	scratch
	C ebx	scratch
	C ecx	counter, negative
	C edx	carry
	C esi	&src[size]
	C edi	dst[2*size-2]
	C ebp

	movl	(%esi,%ecx,4), %eax
	movl	%edx, %ebx

	mull	%eax

	addl	%ebx, 4(%edi,%ecx,8)
	adcl	%eax, 8(%edi,%ecx,8)
	adcl	$0, %edx

	incl	%ecx
	jnz	L(diag)


	movl	SAVE_ESI, %esi
	movl	SAVE_EBX, %ebx

	addl	%edx, 4(%edi)		C dst most significant limb

	movl	SAVE_EDI, %edi
	movl	SAVE_EBP, %ebp
	addl	$FRAME, %esp
	ret



C -----------------------------------------------------------------------------
ifdef(`PIC',`
L(pic_calc):
	addl	(%esp), %ecx
	addl	$L(unroll_inner_end)-L(here)-eval(2*CODE_BYTES_PER_LIMB), %ecx
	addl	%edx, %ecx
	ret_internal
')


EPILOGUE()
