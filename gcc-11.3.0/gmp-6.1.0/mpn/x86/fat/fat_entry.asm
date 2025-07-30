dnl  x86 fat binary entrypoints.

dnl  Copyright 2003, 2012, 2014 Free Software Foundation, Inc.

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


dnl  Forcibly disable profiling.
dnl
dnl  The entrypoints and inits are small enough not to worry about, the real
dnl  routines arrived at will have any profiling.  Also, the way the code
dnl  here ends with a jump means we won't work properly with the
dnl  "instrument" profiling scheme anyway.

define(`WANT_PROFILING',no)


	TEXT


dnl  Usage: FAT_ENTRY(name, offset)
dnl
dnl  Emit a fat binary entrypoint function of the given name.  This is the
dnl  normal entry for applications, eg. __gmpn_add_n.
dnl
dnl  The code simply jumps through the function pointer in __gmpn_cpuvec at
dnl  the given "offset" (in bytes).
dnl
dnl  For non-PIC, the jumps are 5 bytes each, aligning them to 8 should be
dnl  fine for all x86s.
dnl
dnl  For PIC, the jumps are 20 bytes each, and are best aligned to 16 to
dnl  ensure at least the first two instructions don't cross a cache line
dnl  boundary.
dnl
dnl  Note the extra `' ahead of PROLOGUE obscures it from the HAVE_NATIVE
dnl  grepping in configure, stopping that code trying to eval something with
dnl  $1 in it.

define(FAT_ENTRY,
m4_assert_numargs(2)
`	ALIGN(ifdef(`PIC',16,8))
`'PROLOGUE($1)dnl
ifdef(`PIC',`dnl
ifdef(`DARWIN',`
	call	L(movl_eip_edx)
	movl	L(___gmpn_cpuvec)$non_lazy_ptr-.(%edx), %edx
	jmp	*m4_empty_if_zero($2)(%edx)
',`dnl
	call	L(movl_eip_edx)
L(entry_here$2):
	addl	$_GLOBAL_OFFSET_TABLE_+[.-L(entry_here$2)], %edx
	movl	GSYM_PREFIX`'__gmpn_cpuvec@GOT(%edx), %edx
	jmp	*m4_empty_if_zero($2)(%edx)
')
',`dnl non-PIC
	jmp	*GSYM_PREFIX`'__gmpn_cpuvec+$2
')
EPILOGUE()
')


dnl  FAT_ENTRY for each CPUVEC_FUNCS_LIST
dnl

define(`CPUVEC_offset',0)
foreach(i,
`FAT_ENTRY(MPN(i),CPUVEC_offset)
define(`CPUVEC_offset',eval(CPUVEC_offset + 4))',
CPUVEC_FUNCS_LIST)

ifdef(`PIC',`
	ALIGN(8)
L(movl_eip_edx):
	movl	(%esp), %edx
	ret_internal
ifdef(`DARWIN',`
	.section	__IMPORT,__pointers,non_lazy_symbol_pointers
L(___gmpn_cpuvec)$non_lazy_ptr:
	.indirect_symbol	___gmpn_cpuvec
	.long	0
	TEXT
')
')


dnl  Usage: FAT_INIT(name, offset)
dnl
dnl  Emit a fat binary initializer function of the given name.  These
dnl  functions are the initial values for the pointers in __gmpn_cpuvec.
dnl
dnl  The code simply calls __gmpn_cpuvec_init, and then jumps back through
dnl  the __gmpn_cpuvec pointer, at the given "offset" (in bytes).
dnl  __gmpn_cpuvec_init will have stored the address of the selected
dnl  implementation there.
dnl
dnl  Only one of these routines will be executed, and only once, since after
dnl  that all the __gmpn_cpuvec pointers go to real routines.  So there's no
dnl  need for anything special here, just something small and simple.  To
dnl  keep code size down, "fat_init" is a shared bit of code, arrived at
dnl  with the offset in %al.  %al is used since the movb instruction is 2
dnl  bytes where %eax would be 4.
dnl
dnl  Note having `PROLOGUE in FAT_INIT obscures that PROLOGUE from the
dnl  HAVE_NATIVE grepping in configure, preventing that code trying to eval
dnl  something with $1 in it.

define(FAT_INIT,
m4_assert_numargs(2)
`PROLOGUE($1)dnl
	movb	$`'$2, %al
	jmp	L(fat_init)
EPILOGUE()
')

L(fat_init):
	C al	__gmpn_cpuvec byte offset

	movzbl	%al, %eax
	pushl	%eax

ifdef(`PIC',`dnl
ifdef(`DARWIN',`
	sub	$8, %esp
	CALL(	__gmpn_cpuvec_init)
	add	$8, %esp
	call	L(movl_eip_edx)
	movl	L(___gmpn_cpuvec)$non_lazy_ptr-.(%edx), %edx
',`dnl
	pushl	%ebx
	call	L(movl_eip_ebx)
L(init_here):
	addl	$_GLOBAL_OFFSET_TABLE_+[.-L(init_here)], %ebx
	CALL(	__gmpn_cpuvec_init)
	movl	GSYM_PREFIX`'__gmpn_cpuvec@GOT(%ebx), %edx
	popl	%ebx
')
	popl	%eax
	jmp	*(%edx,%eax)

L(movl_eip_ebx):
	movl	(%esp), %ebx
	ret_internal
',`dnl non-PIC
	sub	$8, %esp		C needed on Darwin, harmless elsewhere
	CALL(	__gmpn_cpuvec_init)
	add	$8, %esp		C needed on Darwin, harmless elsewhere
	popl	%eax
	jmp	*GSYM_PREFIX`'__gmpn_cpuvec(%eax)
')

dnl  FAT_INIT for each CPUVEC_FUNCS_LIST
dnl

define(`CPUVEC_offset',0)
foreach(i,
`FAT_INIT(MPN(i`'_init),CPUVEC_offset)
define(`CPUVEC_offset',eval(CPUVEC_offset + 4))',
CPUVEC_FUNCS_LIST)



C long __gmpn_cpuid (char dst[12], int id);
C
C This is called only once, so just something simple and compact is fine.

defframe(PARAM_ID,  8)
defframe(PARAM_DST, 4)
deflit(`FRAME',0)

PROLOGUE(__gmpn_cpuid)
	pushl	%esi		FRAME_pushl()
	pushl	%ebx		FRAME_pushl()
	movl	PARAM_ID, %eax
	cpuid
	movl	PARAM_DST, %esi
	movl	%ebx, (%esi)
	movl	%edx, 4(%esi)
	movl	%ecx, 8(%esi)
	popl	%ebx
	popl	%esi
	ret
EPILOGUE()


C int __gmpn_cpuid_available (void);
C
C Return non-zero if the cpuid instruction is available, which means late
C model 80486 and higher.  80386 and early 80486 don't have cpuid.
C
C The test follows Intel AP-485 application note, namely that if bit 21 is
C modifiable then cpuid is supported.  This test is reentrant and thread
C safe, since of course any interrupt or context switch will preserve the
C flags while we're tinkering with them.
C
C This is called only once, so just something simple and compact is fine.

PROLOGUE(__gmpn_cpuid_available)
	pushf
	popl	%ecx		C old flags

	movl	%ecx, %edx
	xorl	$0x200000, %edx
	pushl	%edx
	popf
	pushf
	popl	%edx		C tweaked flags

	movl	$1, %eax
	cmpl	%ecx, %edx
	jne	L(available)
	xorl	%eax, %eax	C not changed, so cpuid not available

L(available):
	ret
EPILOGUE()
ASM_END()
