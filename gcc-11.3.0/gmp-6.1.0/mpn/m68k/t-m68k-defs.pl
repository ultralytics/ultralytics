#! /usr/bin/perl -w

# Copyright 2001, 2003 Free Software Foundation, Inc.
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


# Usage:  perl t-m68k-defs.pl [-t]
#
# Run this in the mpn/m68k source directory to check that m68k-defs.m4 has
# m68k_defbranch()s or m68k_definsn()s for each instruction used in *.asm
# and */*.asm.  Print nothing if everything is ok.  The -t option prints
# some diagnostic traces.

use strict;
use Getopt::Std;

my %opt;
getopts('t', \%opt);

my %branch;
my %insn;

open(FD, "<m68k-defs.m4")
    or die "Cannot open m68k-defs.m4: $!\nIs this the mpn/m68k source directory?\n";
my ($srcdir, $top_srcdir);
while (<FD>) {
    if (/^m68k_defbranch\(\s*(.*)\)/) { $branch{"b".$1} = 1; }
    if (/^m68k_definsn\(\s*(.*),\s*(.*)\)/) { $insn{$1.$2} = 1; }
}
close(FD);

print "branches: ", join(" ",keys(%branch)), "\n" if $opt{'t'};
print "insns: ", join(" ",keys(%insn)), "\n" if $opt{'t'};


foreach my $file (glob("*.asm"), glob("*/*.asm")) {
    print "file $file\n" if $opt{'t'};

    open(FD, "<$file") or die "Cannot open $file: $!";
    while (<FD>) {
	if (/^[ \t]*C/) { next; };
	if (/^\t([a-z0-9]+)/) {
	    my $opcode = $1;
	    print "opcode $1\n" if $opt{'t'};

	    # instructions with an l, w or b suffix should have a definsn
	    # (unless they're already a defbranch)
	    if ($opcode =~ /[lwb]$/
		&& ! defined $insn{$opcode}
		&& ! defined $branch{$opcode})
	    {
		print "$file: $.: missing m68k_definsn: $opcode\n";
	    }

	    # instructions bXX should have a defbranch (unless they're
	    # already a definsn)
	    if ($opcode =~ /^b/
		&& ! defined $insn{$opcode}
		&& ! defined $branch{$opcode})
	    {
		print "$file: $.: missing m68k_defbranch: $opcode\n";
	    }
	}
    }
    close(FD);
}
