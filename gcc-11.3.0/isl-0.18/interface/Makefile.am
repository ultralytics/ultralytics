AUTOMAKE_OPTIONS = nostdinc

noinst_PROGRAMS = extract_interface

AM_CXXFLAGS = $(CLANG_CXXFLAGS)
AM_LDFLAGS = $(CLANG_LDFLAGS)

includes = -I$(top_builddir) -I$(top_srcdir) \
	-I$(top_builddir)/include -I$(top_srcdir)/include

extract_interface_CPPFLAGS = $(includes)
extract_interface_SOURCES = \
	python.h \
	python.cc \
	extract_interface.h \
	extract_interface.cc
extract_interface_LDADD = \
	-lclangFrontend -lclangSerialization -lclangParse -lclangSema \
	$(LIB_CLANG_EDIT) \
	-lclangAnalysis -lclangAST -lclangLex -lclangBasic -lclangDriver \
	$(CLANG_LIBS) $(CLANG_LDFLAGS)

CLEANFILES = isl.py

test: extract_interface
	./extract_interface$(EXEEXT) $(includes) $(srcdir)/all.h

isl.py: extract_interface isl.py.top
	(cat $(srcdir)/isl.py.top; \
		./extract_interface$(EXEEXT) $(includes) $(srcdir)/all.h) \
			> isl.py

dist-hook: isl.py
	cp isl.py $(distdir)/
