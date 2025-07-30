dnl  AMD K7 mpn_sqr_basecase -- square an mpn number.

dnl  Copyright 1999-2002 Free Software Foundation, Inc.

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


C K7: approx 2.3 cycles/crossproduct, or 4.55 cycles/triangular product
C     (measured on the speed difference between 25 and 50 limbs, which is
C     roughly the Karatsuba recursing range).


dnl  These are the same as mpn/x86/k6/sqr_basecase.asm, see that code for
dnl  some comments.

deflit(SQR_TOOM2_THRESHOLD_MAX, 66)

ifdef(`SQR_TOOM2_THRESHOLD_OVERRIDE',
`define(`SQR_TOOM2_THRESHOLD',SQR_TOOM2_THRESHOLD_OVERRIDE)')

m4_config_gmp_mparam(`SQR_TOOM2_THRESHOLD')
deflit(UNROLL_COUNT, eval(SQR_TOOM2_THRESHOLD-3))


C void mpn_sqr_basecase (mp_ptr dst, mp_srcptr src, mp_size_t size);
C
C With a SQR_TOOM2_THRESHOLD around 50 this code is about 1500 bytes,
C which is quite a bit, but is considered good value since squares big
C enough to use most of the code will be spending quite a few cycles in it.


defframe(PARAM_SIZE,12)
defframe(PARAM_SRC, 8)
defframe(PARAM_DST, 4)

	TEXT
	ALIGN(32)
PROLOGUE(mpn_sqr_basecase)
deflit(`FRAME',0)

	movl	PARAM_SIZE, %ecx
	movl	PARAM_SRC, %eax
	cmpl	$2, %ecx

	movl	PARAM_DST, %edx
	je	L(two_limbs)
	ja	L(three_or_more)


C------------------------------------------------------------------------------
C one limb only
	C eax	src
	C ecx	size
	C edx	dst

	movl	(%eax), %eax
	movl	%edx, %ecx

	mull	%eax

	movl	%edx, 4(%ecx)
	movl	%eax, (%ecx)
	ret


C------------------------------------------------------------------------------
C
C Using the read/modify/write "add"s seems to be faster than saving and
C restoring registers.  Perhaps the loads for the first set hide under the
C mul latency and the second gets store to load forwarding.

	ALIGN(16)
L(two_limbs):
	C eax	src
	C ebx
	C ecx	size
	C edx	dst
deflit(`FRAME',0)

	pushl	%ebx		FRAME_pushl()
	movl	%eax, %ebx	C src
	movl	(%eax), %eax

	movl	%edx, %ecx	C dst

	mull	%eax		C src[0]^2

	movl	%eax, (%ecx)	C dst[0]
	movl	4(%ebx), %eax

	movl	%edx, 4(%ecx)	C dst[1]

	mull	%eax		C src[1]^2

	movl	%eax, 8(%ecx)	C dst[2]
	movl	(%ebx), %eax

	movl	%edx, 12(%ecx)	C dst[3]

	mull	4(%ebx)		C src[0]*src[1]

	popl	%ebx

	addl	%eax, 4(%ecx)
	adcl	%edx, 8(%ecx)
	adcl	$0, 12(%ecx)
	ASSERT(nc)

	addl	%eax, 4(%ecx)
	adcl	%edx, 8(%ecx)
	adcl	$0, 12(%ecx)
	ASSERT(nc)

	ret


C------------------------------------------------------------------------------
defframe(SAVE_EBX,  -4)
defframe(SAVE_ESI,  -8)
defframe(SAVE_EDI, -12)
defframe(SAVE_EBP, -16)
deflit(STACK_SPACE, 16)

L(three_or_more):
	subl	$STACK_SPACE, %esp
	cmpl	$4, %ecx
	jae	L(four_or_more)
deflit(`FRAME',STACK_SPACE)


C------------------------------------------------------------------------------
C Three limbs
C
C Writing out the loads and stores separately at the end of this code comes
C out about 10 cycles faster than using adcls to memory.

	C eax	src
	C ecx	size
	C edx	dst

	movl	%ebx, SAVE_EBX
	movl	%eax, %ebx	C src
	movl	(%eax), %eax

	movl	%edx, %ecx	C dst
	movl	%esi, SAVE_ESI
	movl	%edi, SAVE_EDI

	mull	%eax		C src[0] ^ 2

	movl	%eax, (%ecx)
	movl	4(%ebx), %eax
	movl	%edx, 4(%ecx)

	mull	%eax		C src[1] ^ 2

	movl	%eax, 8(%ecx)
	movl	8(%ebx), %eax
	movl	%edx, 12(%ecx)

	mull	%eax		C src[2] ^ 2

	movl	%eax, 16(%ecx)
	movl	(%ebx), %eax
	movl	%edx, 20(%ecx)

	mull	4(%ebx)		C src[0] * src[1]

	movl	%eax, %esi
	movl	(%ebx), %eax
	movl	%edx, %edi

	mull	8(%ebx)		C src[0] * src[2]

	addl	%eax, %edi
	movl	%ebp, SAVE_EBP
	movl	$0, %ebp

	movl	4(%ebx), %eax
	adcl	%edx, %ebp

	mull	8(%ebx)		C src[1] * src[2]

	xorl	%ebx, %ebx
	addl	%eax, %ebp

	adcl	$0, %edx

	C eax
	C ebx	zero, will be dst[5]
	C ecx	dst
	C edx	dst[4]
	C esi	dst[1]
	C edi	dst[2]
	C ebp	dst[3]

	adcl	$0, %edx
	addl	%esi, %esi

	adcl	%edi, %edi
	movl	4(%ecx), %eax

	adcl	%ebp, %ebp

	adcl	%edx, %edx

	adcl	$0, %ebx
	addl	%eax, %esi
	movl	8(%ecx), %eax

	adcl	%eax, %edi
	movl	12(%ecx), %eax
	movl	%esi, 4(%ecx)

	adcl	%eax, %ebp
	movl	16(%ecx), %eax
	movl	%edi, 8(%ecx)

	movl	SAVE_ESI, %esi
	movl	SAVE_EDI, %edi

	adcl	%eax, %edx
	movl	20(%ecx), %eax
	movl	%ebp, 12(%ecx)

	adcl	%ebx, %eax
	ASSERT(nc)
	movl	SAVE_EBX, %ebx
	movl	SAVE_EBP, %ebp

	movl	%edx, 16(%ecx)
	movl	%eax, 20(%ecx)
	addl	$FRAME, %esp

	ret


C------------------------------------------------------------------------------
L(four_or_more):

C First multiply src[0]*src[1..size-1] and store at dst[1..size].
C Further products are added in rather than stored.

	C eax	src
	C ebx
	C ecx	size
	C edx	dst
	C esi
	C edi
	C ebp

defframe(`VAR_COUNTER',-20)
defframe(`VAR_JMP',    -24)
deflit(EXTRA_STACK_SPACE, 8)

	movl	%ebx, SAVE_EBX
	movl	%edi, SAVE_EDI
	leal	(%edx,%ecx,4), %edi	C &dst[size]

	movl	%esi, SAVE_ESI
	movl	%ebp, SAVE_EBP
	leal	(%eax,%ecx,4), %esi	C &src[size]

	movl	(%eax), %ebp		C multiplier
	movl	$0, %ebx
	decl	%ecx

	negl	%ecx
	subl	$EXTRA_STACK_SPACE, %esp
FRAME_subl_esp(EXTRA_STACK_SPACE)

L(mul_1):
	C eax	scratch
	C ebx	carry
	C ecx	counter
	C edx	scratch
	C esi	&src[size]
	C edi	&dst[size]
	C ebp	multiplier

	movl	(%esi,%ecx,4), %eax

	mull	%ebp

	addl	%ebx, %eax
	movl	%eax, (%edi,%ecx,4)
	movl	$0, %ebx

	adcl	%edx, %ebx
	incl	%ecx
	jnz	L(mul_1)


C Add products src[n]*src[n+1..size-1] at dst[2*n-1...], for each n=1..size-2.
C
C The last two products, which are the bottom right corner of the product
C triangle, are left to the end.  These are src[size-3]*src[size-2,size-1]
C and src[size-2]*src[size-1].  If size is 4 then it's only these corner
C cases that need to be done.
C
C The unrolled code is the same as in mpn_addmul_1, see that routine for
C some comments.
C
C VAR_COUNTER is the outer loop, running from -size+4 to -1, inclusive.
C
C VAR_JMP is the computed jump into the unrolled code, stepped by one code
C chunk each outer loop.
C
C K7 does branch prediction on indirect jumps, which is bad since it's a
C different target each time.  There seems no way to avoid this.

dnl  This value also hard coded in some shifts and adds
deflit(CODE_BYTES_PER_LIMB, 17)

dnl  With the unmodified &src[size] and &dst[size] pointers, the
dnl  displacements in the unrolled code fit in a byte for UNROLL_COUNT
dnl  values up to 31, but above that an offset must be added to them.

deflit(OFFSET,
ifelse(eval(UNROLL_COUNT>31),1,
eval((UNROLL_COUNT-31)*4),
0))

dnl  Because the last chunk of code is generated differently, a label placed
dnl  at the end doesn't work.  Instead calculate the implied end using the
dnl  start and how many chunks of code there are.

deflit(UNROLL_INNER_END,
`L(unroll_inner_start)+eval(UNROLL_COUNT*CODE_BYTES_PER_LIMB)')

	C eax
	C ebx	carry
	C ecx
	C edx
	C esi	&src[size]
	C edi	&dst[size]
	C ebp

	movl	PARAM_SIZE, %ecx
	movl	%ebx, (%edi)

	subl	$4, %ecx
	jz	L(corner)

	negl	%ecx
ifelse(OFFSET,0,,`subl	$OFFSET, %edi')
ifelse(OFFSET,0,,`subl	$OFFSET, %esi')

	movl	%ecx, %edx
	shll	$4, %ecx

ifdef(`PIC',`
	call	L(pic_calc)
L(here):
',`
	leal	UNROLL_INNER_END-eval(2*CODE_BYTES_PER_LIMB)(%ecx,%edx), %ecx
')


	C The calculated jump mustn't come out to before the start of the
	C code available.  This is the limit UNROLL_COUNT puts on the src
	C operand size, but checked here directly using the jump address.
	ASSERT(ae,
	`movl_text_address(L(unroll_inner_start), %eax)
	cmpl	%eax, %ecx')


C------------------------------------------------------------------------------
	ALIGN(16)
L(unroll_outer_top):
	C eax
	C ebx	high limb to store
	C ecx	VAR_JMP
	C edx	VAR_COUNTER, limbs, negative
	C esi	&src[size], constant
	C edi	dst ptr, high of last addmul
	C ebp

	movl	-12+OFFSET(%esi,%edx,4), %ebp	C next multiplier
	movl	-8+OFFSET(%esi,%edx,4), %eax	C first of multiplicand

	movl	%edx, VAR_COUNTER

	mull	%ebp

define(cmovX,`ifelse(eval(UNROLL_COUNT%2),0,`cmovz($@)',`cmovnz($@)')')

	testb	$1, %cl
	movl	%edx, %ebx	C high carry
	movl	%ecx, %edx	C jump

	movl	%eax, %ecx	C low carry
	cmovX(	%ebx, %ecx)	C high carry reverse
	cmovX(	%eax, %ebx)	C low carry reverse

	leal	CODE_BYTES_PER_LIMB(%edx), %eax
	xorl	%edx, %edx
	leal	4(%edi), %edi

	movl	%eax, VAR_JMP

	jmp	*%eax


ifdef(`PIC',`
L(pic_calc):
	addl	(%esp), %ecx
	addl	$UNROLL_INNER_END-eval(2*CODE_BYTES_PER_LIMB)-L(here), %ecx
	addl	%edx, %ecx
	ret_internal
')


	C Must be an even address to preserve the significance of the low
	C bit of the jump address indicating which way around ecx/ebx should
	C start.
	ALIGN(2)

L(unroll_inner_start):
	C eax	next limb
	C ebx	carry high
	C ecx	carry low
	C edx	scratch
	C esi	src
	C edi	dst
	C ebp	multiplier

forloop(`i', UNROLL_COUNT, 1, `
	deflit(`disp_src', eval(-i*4 + OFFSET))
	deflit(`disp_dst', eval(disp_src - 4))

	m4_assert(`disp_src>=-128 && disp_src<128')
	m4_assert(`disp_dst>=-128 && disp_dst<128')

ifelse(eval(i%2),0,`
Zdisp(	movl,	disp_src,(%esi), %eax)
	adcl	%edx, %ebx

	mull	%ebp

Zdisp(  addl,	%ecx, disp_dst,(%edi))
	movl	$0, %ecx

	adcl	%eax, %ebx

',`
	dnl  this bit comes out last
Zdisp(  movl,	disp_src,(%esi), %eax)
	adcl	%edx, %ecx

	mull	%ebp

Zdisp(	addl,	%ebx, disp_dst,(%edi))

ifelse(forloop_last,0,
`	movl	$0, %ebx')

	adcl	%eax, %ecx
')
')

	C eax	next limb
	C ebx	carry high
	C ecx	carry low
	C edx	scratch
	C esi	src
	C edi	dst
	C ebp	multiplier

	adcl	$0, %edx
	addl	%ecx, -4+OFFSET(%edi)
	movl	VAR_JMP, %ecx

	adcl	$0, %edx

	movl	%edx, m4_empty_if_zero(OFFSET) (%edi)
	movl	VAR_COUNTER, %edx

	incl	%edx
	jnz	L(unroll_outer_top)


ifelse(OFFSET,0,,`
	addl	$OFFSET, %esi
	addl	$OFFSET, %edi
')


C------------------------------------------------------------------------------
L(corner):
	C esi	&src[size]
	C edi	&dst[2*size-5]

	movl	-12(%esi), %ebp
	movl	-8(%esi), %eax
	movl	%eax, %ecx

	mull	%ebp

	addl	%eax, -4(%edi)
	movl	-4(%esi), %eax

	adcl	$0, %edx
	movl	%edx, %ebx
	movl	%eax, %esi

	mull	%ebp

	addl	%ebx, %eax

	adcl	$0, %edx
	addl	%eax, (%edi)
	movl	%esi, %eax

	adcl	$0, %edx
	movl	%edx, %ebx

	mull	%ecx

	addl	%ebx, %eax
	movl	%eax, 4(%edi)

	adcl	$0, %edx
	movl	%edx, 8(%edi)



C Left shift of dst[1..2*size-2], high bit shifted out becomes dst[2*size-1].

L(lshift_start):
	movl	PARAM_SIZE, %eax
	movl	PARAM_DST, %edi
	xorl	%ecx, %ecx		C clear carry

	leal	(%edi,%eax,8), %edi
	notl	%eax			C -size-1, preserve carry

	leal	2(%eax), %eax		C -(size-1)

L(lshift):
	C eax	counter, negative
	C ebx
	C ecx
	C edx
	C esi
	C edi	dst, pointing just after last limb
	C ebp

	rcll	-4(%edi,%eax,8)
	rcll	(%edi,%eax,8)
	incl	%eax
	jnz	L(lshift)

	setc	%al

	movl	PARAM_SRC, %esi
	movl	%eax, -4(%edi)		C dst most significant limb

	movl	PARAM_SIZE, %ecx


C Now add in the squares on the diagonal, src[0]^2, src[1]^2, ...,
C src[size-1]^2.  dst[0] hasn't yet been set at all yet, and just gets the
C low limb of src[0]^2.

	movl	(%esi), %eax		C src[0]

	mull	%eax

	leal	(%esi,%ecx,4), %esi	C src point just after last limb
	negl	%ecx

	movl	%eax, (%edi,%ecx,8)	C dst[0]
	incl	%ecx

L(diag):
	C eax	scratch
	C ebx	scratch
	C ecx	counter, negative
	C edx	carry
	C esi	src just after last limb
	C edi	dst just after last limb
	C ebp

	movl	(%esi,%ecx,4), %eax
	movl	%edx, %ebx

	mull	%eax

	addl	%ebx, -4(%edi,%ecx,8)
	adcl	%eax, (%edi,%ecx,8)
	adcl	$0, %edx

	incl	%ecx
	jnz	L(diag)


	movl	SAVE_ESI, %esi
	movl	SAVE_EBX, %ebx

	addl	%edx, -4(%edi)		C dst most significant limb
	movl	SAVE_EDI, %edi

	movl	SAVE_EBP, %ebp
	addl	$FRAME, %esp

	ret

EPILOGUE()
