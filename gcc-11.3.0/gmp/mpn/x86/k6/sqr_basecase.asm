dnl  AMD K6 mpn_sqr_basecase -- square an mpn number.

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


C K6: approx 4.7 cycles per cross product, or 9.2 cycles per triangular
C     product (measured on the speed difference between 17 and 33 limbs,
C     which is roughly the Karatsuba recursing range).


dnl  SQR_TOOM2_THRESHOLD_MAX is the maximum SQR_TOOM2_THRESHOLD this
dnl  code supports.  This value is used only by the tune program to know
dnl  what it can go up to.  (An attempt to compile with a bigger value will
dnl  trigger some m4_assert()s in the code, making the build fail.)
dnl
dnl  The value is determined by requiring the displacements in the unrolled
dnl  addmul to fit in single bytes.  This means a maximum UNROLL_COUNT of
dnl  63, giving a maximum SQR_TOOM2_THRESHOLD of 66.

deflit(SQR_TOOM2_THRESHOLD_MAX, 66)


dnl  Allow a value from the tune program to override config.m4.

ifdef(`SQR_TOOM2_THRESHOLD_OVERRIDE',
`define(`SQR_TOOM2_THRESHOLD',SQR_TOOM2_THRESHOLD_OVERRIDE)')


dnl  UNROLL_COUNT is the number of code chunks in the unrolled addmul.  The
dnl  number required is determined by SQR_TOOM2_THRESHOLD, since
dnl  mpn_sqr_basecase only needs to handle sizes < SQR_TOOM2_THRESHOLD.
dnl
dnl  The first addmul is the biggest, and this takes the second least
dnl  significant limb and multiplies it by the third least significant and
dnl  up.  Hence for a maximum operand size of SQR_TOOM2_THRESHOLD-1
dnl  limbs, UNROLL_COUNT needs to be SQR_TOOM2_THRESHOLD-3.

m4_config_gmp_mparam(`SQR_TOOM2_THRESHOLD')
deflit(UNROLL_COUNT, eval(SQR_TOOM2_THRESHOLD-3))


C void mpn_sqr_basecase (mp_ptr dst, mp_srcptr src, mp_size_t size);
C
C The algorithm is essentially the same as mpn/generic/sqr_basecase.c, but a
C lot of function call overheads are avoided, especially when the given size
C is small.
C
C The code size might look a bit excessive, but not all of it is executed
C and so won't fill up the code cache.  The 1x1, 2x2 and 3x3 special cases
C clearly apply only to those sizes; mid sizes like 10x10 only need part of
C the unrolled addmul; and big sizes like 35x35 that do need all of it will
C at least be getting value for money, because 35x35 spends something like
C 5780 cycles here.
C
C Different values of UNROLL_COUNT give slightly different speeds, between
C 9.0 and 9.2 c/tri-prod measured on the difference between 17 and 33 limbs.
C This isn't a big difference, but it's presumably some alignment effect
C which if understood could give a simple speedup.

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
	je	L(two_limbs)

	movl	PARAM_DST, %edx
	ja	L(three_or_more)


C -----------------------------------------------------------------------------
C one limb only
	C eax	src
	C ebx
	C ecx	size
	C edx	dst

	movl	(%eax), %eax
	movl	%edx, %ecx

	mull	%eax

	movl	%eax, (%ecx)
	movl	%edx, 4(%ecx)
	ret


C -----------------------------------------------------------------------------
	ALIGN(16)
L(two_limbs):
	C eax	src
	C ebx
	C ecx	size
	C edx	dst

	pushl	%ebx
	movl	%eax, %ebx	C src
deflit(`FRAME',4)

	movl	(%ebx), %eax
	movl	PARAM_DST, %ecx

	mull	%eax		C src[0]^2

	movl	%eax, (%ecx)
	movl	4(%ebx), %eax

	movl	%edx, 4(%ecx)

	mull	%eax		C src[1]^2

	movl	%eax, 8(%ecx)
	movl	(%ebx), %eax

	movl	%edx, 12(%ecx)
	movl	4(%ebx), %edx

	mull	%edx		C src[0]*src[1]

	addl	%eax, 4(%ecx)

	adcl	%edx, 8(%ecx)
	adcl	$0, 12(%ecx)

	popl	%ebx
	addl	%eax, 4(%ecx)

	adcl	%edx, 8(%ecx)
	adcl	$0, 12(%ecx)

	ret


C -----------------------------------------------------------------------------
L(three_or_more):
deflit(`FRAME',0)
	cmpl	$4, %ecx
	jae	L(four_or_more)


C -----------------------------------------------------------------------------
C three limbs
	C eax	src
	C ecx	size
	C edx	dst

	pushl	%ebx
	movl	%eax, %ebx	C src

	movl	(%ebx), %eax
	movl	%edx, %ecx	C dst

	mull	%eax		C src[0] ^ 2

	movl	%eax, (%ecx)
	movl	4(%ebx), %eax

	movl	%edx, 4(%ecx)
	pushl	%esi

	mull	%eax		C src[1] ^ 2

	movl	%eax, 8(%ecx)
	movl	8(%ebx), %eax

	movl	%edx, 12(%ecx)
	pushl	%edi

	mull	%eax		C src[2] ^ 2

	movl	%eax, 16(%ecx)
	movl	(%ebx), %eax

	movl	%edx, 20(%ecx)
	movl	4(%ebx), %edx

	mull	%edx		C src[0] * src[1]

	movl	%eax, %esi
	movl	(%ebx), %eax

	movl	%edx, %edi
	movl	8(%ebx), %edx

	pushl	%ebp
	xorl	%ebp, %ebp

	mull	%edx		C src[0] * src[2]

	addl	%eax, %edi
	movl	4(%ebx), %eax

	adcl	%edx, %ebp

	movl	8(%ebx), %edx

	mull	%edx		C src[1] * src[2]

	addl	%eax, %ebp

	adcl	$0, %edx


	C eax	will be dst[5]
	C ebx
	C ecx	dst
	C edx	dst[4]
	C esi	dst[1]
	C edi	dst[2]
	C ebp	dst[3]

	xorl	%eax, %eax
	addl	%esi, %esi
	adcl	%edi, %edi
	adcl	%ebp, %ebp
	adcl	%edx, %edx
	adcl	$0, %eax

	addl	%esi, 4(%ecx)
	adcl	%edi, 8(%ecx)
	adcl	%ebp, 12(%ecx)

	popl	%ebp
	popl	%edi

	adcl	%edx, 16(%ecx)

	popl	%esi
	popl	%ebx

	adcl	%eax, 20(%ecx)
	ASSERT(nc)

	ret


C -----------------------------------------------------------------------------

defframe(SAVE_EBX,   -4)
defframe(SAVE_ESI,   -8)
defframe(SAVE_EDI,   -12)
defframe(SAVE_EBP,   -16)
defframe(VAR_COUNTER,-20)
defframe(VAR_JMP,    -24)
deflit(STACK_SPACE, 24)

	ALIGN(16)
L(four_or_more):

	C eax	src
	C ebx
	C ecx	size
	C edx	dst
	C esi
	C edi
	C ebp

C First multiply src[0]*src[1..size-1] and store at dst[1..size].
C
C A test was done calling mpn_mul_1 here to get the benefit of its unrolled
C loop, but this was only a tiny speedup; at 35 limbs it took 24 cycles off
C a 5780 cycle operation, which is not surprising since the loop here is 8
C c/l and mpn_mul_1 is 6.25 c/l.

	subl	$STACK_SPACE, %esp	deflit(`FRAME',STACK_SPACE)

	movl	%edi, SAVE_EDI
	leal	4(%edx), %edi

	movl	%ebx, SAVE_EBX
	leal	4(%eax), %ebx

	movl	%esi, SAVE_ESI
	xorl	%esi, %esi

	movl	%ebp, SAVE_EBP

	C eax
	C ebx	src+4
	C ecx	size
	C edx
	C esi
	C edi	dst+4
	C ebp

	movl	(%eax), %ebp	C multiplier
	leal	-1(%ecx), %ecx	C size-1, and pad to a 16 byte boundary


	ALIGN(16)
L(mul_1):
	C eax	scratch
	C ebx	src ptr
	C ecx	counter
	C edx	scratch
	C esi	carry
	C edi	dst ptr
	C ebp	multiplier

	movl	(%ebx), %eax
	addl	$4, %ebx

	mull	%ebp

	addl	%esi, %eax
	movl	$0, %esi

	adcl	%edx, %esi

	movl	%eax, (%edi)
	addl	$4, %edi

	loop	L(mul_1)


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
C
C K6 doesn't do any branch prediction on indirect jumps, which is good
C actually because it's a different target each time.  The unrolled addmul
C is about 3 cycles/limb faster than a simple loop, so the 6 cycle cost of
C the indirect jump is quickly recovered.


dnl  This value is also implicitly encoded in a shift and add.
dnl
deflit(CODE_BYTES_PER_LIMB, 15)

dnl  With the unmodified &src[size] and &dst[size] pointers, the
dnl  displacements in the unrolled code fit in a byte for UNROLL_COUNT
dnl  values up to 31.  Above that an offset must be added to them.
dnl
deflit(OFFSET,
ifelse(eval(UNROLL_COUNT>31),1,
eval((UNROLL_COUNT-31)*4),
0))

	C eax
	C ebx	&src[size]
	C ecx
	C edx
	C esi	carry
	C edi	&dst[size]
	C ebp

	movl	PARAM_SIZE, %ecx
	movl	%esi, (%edi)

	subl	$4, %ecx
	jz	L(corner)

	movl	%ecx, %edx
ifelse(OFFSET,0,,
`	subl	$OFFSET, %ebx')

	shll	$4, %ecx
ifelse(OFFSET,0,,
`	subl	$OFFSET, %edi')

	negl	%ecx

ifdef(`PIC',`
	call	L(pic_calc)
L(here):
',`
	leal	L(unroll_inner_end)-eval(2*CODE_BYTES_PER_LIMB)(%ecx,%edx), %ecx
')
	negl	%edx


	C The calculated jump mustn't be before the start of the available
	C code.  This is the limitation UNROLL_COUNT puts on the src operand
	C size, but checked here using the jump address directly.
	C
	ASSERT(ae,`
	movl_text_address( L(unroll_inner_start), %eax)
	cmpl	%eax, %ecx
	')


C -----------------------------------------------------------------------------
	ALIGN(16)
L(unroll_outer_top):
	C eax
	C ebx	&src[size], constant
	C ecx	VAR_JMP
	C edx	VAR_COUNTER, limbs, negative
	C esi	high limb to store
	C edi	dst ptr, high of last addmul
	C ebp

	movl	-12+OFFSET(%ebx,%edx,4), %ebp	C multiplier
	movl	%edx, VAR_COUNTER

	movl	-8+OFFSET(%ebx,%edx,4), %eax	C first limb of multiplicand

	mull	%ebp

	testb	$1, %cl

	movl	%edx, %esi	C high carry
	movl	%ecx, %edx	C jump

	movl	%eax, %ecx	C low carry
	leal	CODE_BYTES_PER_LIMB(%edx), %edx

	movl	%edx, VAR_JMP
	leal	4(%edi), %edi

	C A branch-free version of this using some xors was found to be a
	C touch slower than just a conditional jump, despite the jump
	C switching between taken and not taken on every loop.

ifelse(eval(UNROLL_COUNT%2),0,
	jz,jnz)	L(unroll_noswap)
	movl	%esi, %eax	C high,low carry other way around

	movl	%ecx, %esi
	movl	%eax, %ecx
L(unroll_noswap):

	jmp	*%edx


	C Must be on an even address here so the low bit of the jump address
	C will indicate which way around ecx/esi should start.
	C
	C An attempt was made at padding here to get the end of the unrolled
	C code to come out on a good alignment, to save padding before
	C L(corner).  This worked, but turned out to run slower than just an
	C ALIGN(2).  The reason for this is not clear, it might be related
	C to the different speeds on different UNROLL_COUNTs noted above.

	ALIGN(2)

L(unroll_inner_start):
	C eax	scratch
	C ebx	src
	C ecx	carry low
	C edx	scratch
	C esi	carry high
	C edi	dst
	C ebp	multiplier
	C
	C 15 code bytes each limb
	C ecx/esi swapped on each chunk

forloop(`i', UNROLL_COUNT, 1, `
	deflit(`disp_src', eval(-i*4 + OFFSET))
	deflit(`disp_dst', eval(disp_src - 4))

	m4_assert(`disp_src>=-128 && disp_src<128')
	m4_assert(`disp_dst>=-128 && disp_dst<128')

ifelse(eval(i%2),0,`
Zdisp(	movl,	disp_src,(%ebx), %eax)
	mull	%ebp
Zdisp(	addl,	%esi, disp_dst,(%edi))
	adcl	%eax, %ecx
	movl	%edx, %esi
	jadcl0( %esi)
',`
	dnl  this one comes out last
Zdisp(	movl,	disp_src,(%ebx), %eax)
	mull	%ebp
Zdisp(	addl,	%ecx, disp_dst,(%edi))
	adcl	%eax, %esi
	movl	%edx, %ecx
	jadcl0( %ecx)
')
')
L(unroll_inner_end):

	addl	%esi, -4+OFFSET(%edi)

	movl	VAR_COUNTER, %edx
	jadcl0(	%ecx)

	movl	%ecx, m4_empty_if_zero(OFFSET)(%edi)
	movl	VAR_JMP, %ecx

	incl	%edx
	jnz	L(unroll_outer_top)


ifelse(OFFSET,0,,`
	addl	$OFFSET, %ebx
	addl	$OFFSET, %edi
')


C -----------------------------------------------------------------------------
	ALIGN(16)
L(corner):
	C ebx	&src[size]
	C edi	&dst[2*size-5]

	movl	-12(%ebx), %ebp

	movl	-8(%ebx), %eax
	movl	%eax, %ecx

	mull	%ebp

	addl	%eax, -4(%edi)
	adcl	$0, %edx

	movl	-4(%ebx), %eax
	movl	%edx, %esi
	movl	%eax, %ebx

	mull	%ebp

	addl	%esi, %eax
	adcl	$0, %edx

	addl	%eax, (%edi)
	adcl	$0, %edx

	movl	%edx, %esi
	movl	%ebx, %eax

	mull	%ecx

	addl	%esi, %eax
	movl	%eax, 4(%edi)

	adcl	$0, %edx

	movl	%edx, 8(%edi)


C -----------------------------------------------------------------------------
C Left shift of dst[1..2*size-2], the bit shifted out becomes dst[2*size-1].
C The loop measures about 6 cycles/iteration, though it looks like it should
C decode in 5.

L(lshift_start):
	movl	PARAM_SIZE, %ecx

	movl	PARAM_DST, %edi
	subl	$1, %ecx		C size-1 and clear carry

	movl	PARAM_SRC, %ebx
	movl	%ecx, %edx

	xorl	%eax, %eax		C ready for adcl


	ALIGN(16)
L(lshift):
	C eax
	C ebx	src (for later use)
	C ecx	counter, decrementing
	C edx	size-1 (for later use)
	C esi
	C edi	dst, incrementing
	C ebp

	rcll	4(%edi)
	rcll	8(%edi)
	leal	8(%edi), %edi
	loop	L(lshift)


	adcl	%eax, %eax

	movl	%eax, 4(%edi)		C dst most significant limb
	movl	(%ebx), %eax		C src[0]

	leal	4(%ebx,%edx,4), %ebx	C &src[size]
	subl	%edx, %ecx		C -(size-1)


C -----------------------------------------------------------------------------
C Now add in the squares on the diagonal, src[0]^2, src[1]^2, ...,
C src[size-1]^2.  dst[0] hasn't yet been set at all yet, and just gets the
C low limb of src[0]^2.


	mull	%eax

	movl	%eax, (%edi,%ecx,8)	C dst[0]


	ALIGN(16)
L(diag):
	C eax	scratch
	C ebx	&src[size]
	C ecx	counter, negative
	C edx	carry
	C esi	scratch
	C edi	dst[2*size-2]
	C ebp

	movl	(%ebx,%ecx,4), %eax
	movl	%edx, %esi

	mull	%eax

	addl	%esi, 4(%edi,%ecx,8)
	adcl	%eax, 8(%edi,%ecx,8)
	adcl	$0, %edx

	incl	%ecx
	jnz	L(diag)


	movl	SAVE_EBX, %ebx
	movl	SAVE_ESI, %esi

	addl	%edx, 4(%edi)		C dst most significant limb

	movl	SAVE_EDI, %edi
	movl	SAVE_EBP, %ebp
	addl	$FRAME, %esp
	ret



C -----------------------------------------------------------------------------
ifdef(`PIC',`
L(pic_calc):
	C See mpn/x86/README about old gas bugs
	addl	(%esp), %ecx
	addl	$L(unroll_inner_end)-L(here)-eval(2*CODE_BYTES_PER_LIMB), %ecx
	addl	%edx, %ecx
	ret_internal
')


EPILOGUE()
