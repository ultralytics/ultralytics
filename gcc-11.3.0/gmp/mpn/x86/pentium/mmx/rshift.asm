dnl  Intel P5 mpn_rshift -- mpn right shift.

dnl  Copyright 2000, 2002 Free Software Foundation, Inc.

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


C P5: 1.75 cycles/limb.


C mp_limb_t mpn_rshift (mp_ptr dst, mp_srcptr src, mp_size_t size,
C                       unsigned shift);
C
C Shift src,size right by shift many bits and store the result in dst,size.
C Zeros are shifted in at the left.  Return the bits shifted out at the
C right.
C
C It takes 6 mmx instructions to process 2 limbs, making 1.5 cycles/limb,
C and with a 4 limb loop and 1 cycle of loop overhead the total is 1.75 c/l.
C
C Full speed depends on source and destination being aligned.  Unaligned mmx
C loads and stores on P5 don't pair and have a 2 cycle penalty.  Some hairy
C setups and finish-ups are done to ensure alignment for the loop.
C
C MMX shifts work out a bit faster even for the simple loop.

defframe(PARAM_SHIFT,16)
defframe(PARAM_SIZE, 12)
defframe(PARAM_SRC,  8)
defframe(PARAM_DST,  4)
deflit(`FRAME',0)

dnl  Minimum 5, because the unrolled loop can't handle less.
deflit(UNROLL_THRESHOLD, 5)

	TEXT
	ALIGN(8)

PROLOGUE(mpn_rshift)

	pushl	%ebx
	pushl	%edi
deflit(`FRAME',8)

	movl	PARAM_SIZE, %eax
	movl	PARAM_DST, %edx

	movl	PARAM_SRC, %ebx
	movl	PARAM_SHIFT, %ecx

	cmp	$UNROLL_THRESHOLD, %eax
	jae	L(unroll)

	decl	%eax
	movl	(%ebx), %edi		C src low limb

	jnz	L(simple)

	shrdl(	%cl, %edi, %eax)	C eax was decremented to zero

	shrl	%cl, %edi

	movl	%edi, (%edx)		C dst low limb
	popl	%edi			C risk of data cache bank clash

	popl	%ebx

	ret


C -----------------------------------------------------------------------------
	ALIGN(8)
L(simple):
	C eax	size-1
	C ebx	src
	C ecx	shift
	C edx	dst
	C esi
	C edi
	C ebp
deflit(`FRAME',8)

	movd	(%ebx), %mm5		C src[0]
	leal	(%ebx,%eax,4), %ebx	C &src[size-1]

	movd	%ecx, %mm6		C rshift
	leal	-4(%edx,%eax,4), %edx	C &dst[size-2]

	psllq	$32, %mm5
	negl	%eax


C This loop is 5 or 8 cycles, with every second load unaligned and a wasted
C cycle waiting for the mm0 result to be ready.  For comparison a shrdl is 4
C cycles and would be 8 in a simple loop.  Using mmx helps the return value
C and last limb calculations too.

L(simple_top):
	C eax	counter, limbs, negative
	C ebx	&src[size-1]
	C ecx	return value
	C edx	&dst[size-2]
	C
	C mm0	scratch
	C mm5	return value
	C mm6	shift

	movq	(%ebx,%eax,4), %mm0
	incl	%eax

	psrlq	%mm6, %mm0

	movd	%mm0, (%edx,%eax,4)
	jnz	L(simple_top)


	movd	(%ebx), %mm0
	psrlq	%mm6, %mm5		C return value

	psrlq	%mm6, %mm0
	popl	%edi

	movd	%mm5, %eax
	popl	%ebx

	movd	%mm0, 4(%edx)

	emms

	ret


C -----------------------------------------------------------------------------
	ALIGN(8)
L(unroll):
	C eax	size
	C ebx	src
	C ecx	shift
	C edx	dst
	C esi
	C edi
	C ebp
deflit(`FRAME',8)

	movd	(%ebx), %mm5		C src[0]
	movl	$4, %edi

	movd	%ecx, %mm6		C rshift
	testl	%edi, %ebx

	psllq	$32, %mm5
	jz	L(start_src_aligned)


	C src isn't aligned, process low limb separately (marked xxx) and
	C step src and dst by one limb, making src aligned.
	C
	C source                  ebx
	C --+-------+-------+-------+
	C           |          xxx  |
	C --+-------+-------+-------+
	C         4mod8   0mod8   4mod8
	C
	C         dest            edx
	C         --+-------+-------+
	C           |       |  xxx  |
	C         --+-------+-------+

	movq	(%ebx), %mm0		C unaligned load

	psrlq	%mm6, %mm0
	addl	$4, %ebx

	decl	%eax

	movd	%mm0, (%edx)
	addl	$4, %edx
L(start_src_aligned):


	movq	(%ebx), %mm1
	testl	%edi, %edx

	psrlq	%mm6, %mm5		C retval
	jz	L(start_dst_aligned)

	C dst isn't aligned, add 4 to make it so, and pretend the shift is
	C 32 bits extra.  Low limb of dst (marked xxx) handled here
	C separately.
	C
	C          source          ebx
	C          --+-------+-------+
	C            |      mm1      |
	C          --+-------+-------+
	C                  4mod8   0mod8
	C
	C  dest                    edx
	C  --+-------+-------+-------+
	C                    |  xxx  |
	C  --+-------+-------+-------+
	C          4mod8   0mod8   4mod8

	movq	%mm1, %mm0
	addl	$32, %ecx		C new shift

	psrlq	%mm6, %mm0

	movd	%ecx, %mm6

	movd	%mm0, (%edx)
	addl	$4, %edx
L(start_dst_aligned):


	movq	8(%ebx), %mm3
	negl	%ecx

	movq	%mm3, %mm2		C mm2 src qword
	addl	$64, %ecx

	movd	%ecx, %mm7
	psrlq	%mm6, %mm1

	leal	-12(%ebx,%eax,4), %ebx
	leal	-20(%edx,%eax,4), %edx

	psllq	%mm7, %mm3
	subl	$7, %eax		C size-7

	por	%mm1, %mm3		C mm3 ready to store
	negl	%eax			C -(size-7)

	jns	L(finish)


	C This loop is the important bit, the rest is just support.  Careful
	C instruction scheduling achieves the claimed 1.75 c/l.  The
	C relevant parts of the pairing rules are:
	C
	C - mmx loads and stores execute only in the U pipe
	C - only one mmx shift in a pair
	C - wait one cycle before storing an mmx register result
	C - the usual address generation interlock
	C
	C Two qword calculations are slightly interleaved.  The instructions
	C marked "C" belong to the second qword, and the "C prev" one is for
	C the second qword from the previous iteration.

	ALIGN(8)
L(unroll_loop):
	C eax	counter, limbs, negative
	C ebx	&src[size-12]
	C ecx
	C edx	&dst[size-12]
	C esi
	C edi
	C
	C mm0
	C mm1
	C mm2	src qword from -8(%ebx,%eax,4)
	C mm3	dst qword ready to store to -8(%edx,%eax,4)
	C
	C mm5	return value
	C mm6	rshift
	C mm7	lshift

	movq	(%ebx,%eax,4), %mm0
	psrlq	%mm6, %mm2

	movq	%mm0, %mm1
	psllq	%mm7, %mm0

	movq	%mm3, -8(%edx,%eax,4)	C prev
	por	%mm2, %mm0

	movq	8(%ebx,%eax,4), %mm3	C
	psrlq	%mm6, %mm1		C

	movq	%mm0, (%edx,%eax,4)
	movq	%mm3, %mm2		C

	psllq	%mm7, %mm3		C
	addl	$4, %eax

	por	%mm1, %mm3		C
	js	L(unroll_loop)


L(finish):
	C eax	0 to 3 representing respectively 3 to 0 limbs remaining

	testb	$2, %al

	jnz	L(finish_no_two)

	movq	(%ebx,%eax,4), %mm0
	psrlq	%mm6, %mm2

	movq	%mm0, %mm1
	psllq	%mm7, %mm0

	movq	%mm3, -8(%edx,%eax,4)	C prev
	por	%mm2, %mm0

	movq	%mm1, %mm2
	movq	%mm0, %mm3

	addl	$2, %eax
L(finish_no_two):


	C eax	2 or 3 representing respectively 1 or 0 limbs remaining
	C
	C mm2	src prev qword, from -8(%ebx,%eax,4)
	C mm3	dst qword, for -8(%edx,%eax,4)

	testb	$1, %al
	popl	%edi

	movd	%mm5, %eax	C retval
	jnz	L(finish_zero)


	C One extra limb, destination was aligned.
	C
	C source                ebx
	C +-------+---------------+--
	C |       |      mm2      |
	C +-------+---------------+--
	C
	C dest                                  edx
	C +-------+---------------+---------------+--
	C |       |               |      mm3      |
	C +-------+---------------+---------------+--
	C
	C mm6 = shift
	C mm7 = ecx = 64-shift


	C One extra limb, destination was unaligned.
	C
	C source                ebx
	C +-------+---------------+--
	C |       |      mm2      |
	C +-------+---------------+--
	C
	C dest                          edx
	C +---------------+---------------+--
	C |               |      mm3      |
	C +---------------+---------------+--
	C
	C mm6 = shift+32
	C mm7 = ecx = 64-(shift+32)


	C In both cases there's one extra limb of src to fetch and combine
	C with mm2 to make a qword at 8(%edx), and in the aligned case
	C there's a further extra limb of dst to be formed.


	movd	8(%ebx), %mm0
	psrlq	%mm6, %mm2

	movq	%mm0, %mm1
	psllq	%mm7, %mm0

	movq	%mm3, (%edx)
	por	%mm2, %mm0

	psrlq	%mm6, %mm1
	andl	$32, %ecx

	popl	%ebx
	jz	L(finish_one_unaligned)

	C dst was aligned, must store one extra limb
	movd	%mm1, 16(%edx)
L(finish_one_unaligned):

	movq	%mm0, 8(%edx)

	emms

	ret


L(finish_zero):

	C No extra limbs, destination was aligned.
	C
	C source        ebx
	C +---------------+--
	C |      mm2      |
	C +---------------+--
	C
	C dest                        edx+4
	C +---------------+---------------+--
	C |               |      mm3      |
	C +---------------+---------------+--
	C
	C mm6 = shift
	C mm7 = ecx = 64-shift


	C No extra limbs, destination was unaligned.
	C
	C source        ebx
	C +---------------+--
	C |      mm2      |
	C +---------------+--
	C
	C dest                edx+4
	C +-------+---------------+--
	C |       |      mm3      |
	C +-------+---------------+--
	C
	C mm6 = shift+32
	C mm7 = 64-(shift+32)


	C The movd for the unaligned case is clearly the same data as the
	C movq for the aligned case, it's just a choice between whether one
	C or two limbs should be written.


	movq	%mm3, 4(%edx)
	psrlq	%mm6, %mm2

	movd	%mm2, 12(%edx)
	andl	$32, %ecx

	popl	%ebx
	jz	L(finish_zero_unaligned)

	movq	%mm2, 12(%edx)
L(finish_zero_unaligned):

	emms

	ret

EPILOGUE()
