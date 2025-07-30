#! /usr/bin/perl

# Copyright 2000, 2001 Free Software Foundation, Inc.
#
#  This file is part of the GNU MP Library.
#
#  The GNU MP Library is free software; you can redistribute it and/or modify
#  it under the terms of either:
#
#    * the GNU Lesser General Public License as published by the Free
#      Software Foundation; either version 3 of the License, or (at your
#      option) any later version.
#
#  or
#
#    * the GNU General Public License as published by the Free Software
#      Foundation; either version 2 of the License, or (at your option) any
#      later version.
#
#  or both in parallel, as here.
#
#  The GNU MP Library is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
#  or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
#  for more details.
#
#  You should have received copies of the GNU General Public License and the
#  GNU Lesser General Public License along with the GNU MP Library.  If not,
#  see https://www.gnu.org/licenses/.


# Usage: cross.pl [filename.o]...
#
# Produce an annotated disassembly of the given object files, indicating
# certain code alignment and addressing mode problems afflicting K6 chips.
# "ZZ" is used on all annotations, so this can be searched for.
#
# With no arguments, all .o files corresponding to .asm files are processed.
# This is good in the mpn object directory of a k6*-*-* build.
#
# Code alignments of 8 bytes or more are handled.  When 32 is used, cache
# line boundaries will fall in at offsets 0x20,0x40,etc and problems are
# flagged at those locations.  When 16 is used, the line boundaries can also
# fall at offsets 0x10,0x30,0x50,etc, depending where the file is loaded, so
# problems are identified there too.  Likewise when 8 byte alignment is used
# problems are flagged additionally at 0x08,0x18,0x28,etc.
#
# Usually 32 byte alignment is used for k6 routines, but less is certainly
# possible if through good luck, or a little tweaking, cache line crossing
# problems can be avoided at the extra locations.
#
# Bugs:
#
# Instructions without mod/rm bytes or which are already vector decoded are
# unaffected by cache line boundary crossing, but not all of these have yet
# been put in as exceptions.  All that occur in practice in GMP are present
# though.
#
# There's no messages for using the vector decoded addressing mode (%esi),
# but that's easy to avoid when coding.
#
# Future:
#
# Warn about jump targets that are poorly aligned (less than 2 instructions
# before a cache line boundary).

use strict;

sub disassemble {
    my ($file) = @_;
    my ($addr,$b1,$b2,$b3, $prefix,$opcode,$modrm);
    my $align;

    open (IN, "objdump -Srfh $file |")
	|| die "Cannot open pipe from objdump\n";
    while (<IN>) {
	print;

	if (/^[ \t]*[0-9]+[ \t]+\.text[ \t]/ && /2\*\*([0-9]+)$/) {
	    $align = 1 << $1;
	    if ($align < 8) {
		print "ZZ cross.pl cannot handle alignment < 2**3\n";
		$align = 8
	    }
	}

	if (/^[ \t]*([0-9a-f]*):[ \t]*([0-9a-f]+)[ \t]+([0-9a-f]+)[ \t]+([0-9a-f]+)/) {
	    ($addr,$b1,$b2,$b3) = ($1,$2,$3,$4);

	} elsif (/^[ \t]*([0-9a-f]*):[ \t]*([0-9a-f]+)[ \t]+([0-9a-f]+)/) {
	    ($addr,$b1,$b2,$b3) = ($1,$2,$3,'');

	} elsif (/^[ \t]*([0-9a-f]*):[ \t]*([0-9a-f]+)/) {
	    ($addr,$b1,$b2,$b3) = ($1,$2,'','');

	} else {
	    next;
	}

	if ($b1 =~ /0f/) {
	    $prefix = $b1;
	    $opcode = $b2;
	    $modrm = $b3;
	} else {
	    $prefix = '';
	    $opcode = $b1;
	    $modrm = $b2;
	}

	# modrm of the form 00-xxx-100 with an 0F prefix is the problem case
	# for K6 and pre-CXT K6-2
	if ($prefix =~ /0f/
	    && $opcode !~ /^8/         # jcond disp32
	    && $modrm =~ /^[0-3][4c]/) {
	    print "ZZ ($file) >3 bytes to determine instruction length [K6]\n";
	}

	# with just an opcode, starting 1f mod 20h
	if (($align==32 && $addr =~ /[13579bdf]f$/
	     || $align==16 && $addr =~ /f$/
	     || $align==8 && $addr =~ /[7f]$/)
	    && $prefix !~ /0f/
	    && $opcode !~ /1[012345]/ # adc
	    && $opcode !~ /1[89abcd]/ # sbb
	    && $opcode !~ /^4/        # inc/dec reg
	    && $opcode !~ /^5/        # push/pop reg
	    && $opcode !~ /68/        # push $imm32
	    && $opcode !~ /^7/        # jcond disp8
	    && $opcode !~ /a[89]/     # test+imm
	    && $opcode !~ /a[a-f]/    # stos/lods/scas
	    && $opcode !~ /b8/        # movl $imm32,%eax
	    && $opcode !~ /d[0123]/   # rcl
	    && $opcode !~ /e[0123]/   # loop/loopz/loopnz/jcxz
	    && $opcode !~ /e8/        # call disp32
	    && $opcode !~ /e[9b]/     # jmp disp32/disp8
	    && $opcode !~ /f[89abcd]/ # clc,stc,cli,sti,cld,std
	    && !($opcode =~ /f[67]/          # grp 1
		 && $modrm =~ /^[2367abef]/) # mul, imul, div, idiv
	    && $modrm !~ /^$/) {
	    print "ZZ ($file) opcode/modrm cross 32-byte boundary\n";
	}

	# with an 0F prefix, anything starting at 1f mod 20h
	if (($align==32 && $addr =~ /[13579bdf][f]$/
	     || $align==16 && $addr =~ /f$/
	     || $align==8 && $addr =~ /[7f]$/)
	    && $prefix =~ /0f/
	    && $opcode !~ /af/        # imul
	    && $opcode !~ /a[45]/     # shldl
	    && $opcode !~ /a[cd]/     # shrdl
	    ) {
	    print "ZZ ($file) prefix/opcode cross 32-byte boundary\n";
	}

	# with an 0F prefix, anything with mod/rm starting at 1e mod 20h
	if (($align==32 && $addr =~ /[13579bdf][e]$/
	     || $align==16 && $addr =~ /[e]$/
	     || $align==8 && $addr =~ /[6e]$/)
	    && $prefix =~ /0f/
	     && $opcode !~ /^8/        # jcond disp32
	     && $opcode !~ /af/        # imull reg,reg
	     && $opcode !~ /a[45]/     # shldl
	     && $opcode !~ /a[cd]/     # shrdl
	    && $modrm !~ /^$/) {
	    print "ZZ ($file) prefix/opcode/modrm cross 32-byte boundary\n";
	}
    }
    close IN || die "Error from objdump (or objdump not available)\n";
}


my @files;
if ($#ARGV >= 0) {
    @files = @ARGV;
} else {
    @files = glob "*.asm";
    map {s/.asm/.o/} @files;
}

foreach (@files)  {
    disassemble($_);
}
