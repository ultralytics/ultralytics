/*
 * Copyright 2011 Sven Verdoolaege. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 
 *    1. Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 * 
 *    2. Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 * 
 * THIS SOFTWARE IS PROVIDED BY SVEN VERDOOLAEGE ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SVEN VERDOOLAEGE OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 * The views and conclusions contained in the software and documentation
 * are those of the authors and should not be interpreted as
 * representing official policies, either expressed or implied, of
 * Sven Verdoolaege.
 */ 

#include "isl_config.h"

#include <assert.h>
#include <iostream>
#ifdef HAVE_ADT_OWNINGPTR_H
#include <llvm/ADT/OwningPtr.h>
#else
#include <memory>
#endif
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Host.h>
#include <llvm/Support/ManagedStatic.h>
#include <clang/AST/ASTContext.h>
#include <clang/AST/ASTConsumer.h>
#include <clang/Basic/FileSystemOptions.h>
#include <clang/Basic/FileManager.h>
#include <clang/Basic/TargetOptions.h>
#include <clang/Basic/TargetInfo.h>
#include <clang/Basic/Version.h>
#include <clang/Driver/Compilation.h>
#include <clang/Driver/Driver.h>
#include <clang/Driver/Tool.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/CompilerInvocation.h>
#ifdef HAVE_BASIC_DIAGNOSTICOPTIONS_H
#include <clang/Basic/DiagnosticOptions.h>
#else
#include <clang/Frontend/DiagnosticOptions.h>
#endif
#include <clang/Frontend/TextDiagnosticPrinter.h>
#include <clang/Frontend/Utils.h>
#include <clang/Lex/HeaderSearch.h>
#ifdef HAVE_LEX_PREPROCESSOROPTIONS_H
#include <clang/Lex/PreprocessorOptions.h>
#else
#include <clang/Frontend/PreprocessorOptions.h>
#endif
#include <clang/Lex/Preprocessor.h>
#include <clang/Parse/ParseAST.h>
#include <clang/Sema/Sema.h>

#include "extract_interface.h"
#include "python.h"

using namespace std;
using namespace clang;
using namespace clang::driver;

#ifdef HAVE_ADT_OWNINGPTR_H
#define unique_ptr	llvm::OwningPtr
#endif

static llvm::cl::opt<string> InputFilename(llvm::cl::Positional,
			llvm::cl::Required, llvm::cl::desc("<input file>"));
static llvm::cl::list<string> Includes("I",
			llvm::cl::desc("Header search path"),
			llvm::cl::value_desc("path"), llvm::cl::Prefix);

static const char *ResourceDir =
	CLANG_PREFIX "/lib/clang/" CLANG_VERSION_STRING;

/* Does decl have an attribute of the following form?
 *
 *	__attribute__((annotate("name")))
 */
bool has_annotation(Decl *decl, const char *name)
{
	if (!decl->hasAttrs())
		return false;

	AttrVec attrs = decl->getAttrs();
	for (AttrVec::const_iterator i = attrs.begin() ; i != attrs.end(); ++i) {
		const AnnotateAttr *ann = dyn_cast<AnnotateAttr>(*i);
		if (!ann)
			continue;
		if (ann->getAnnotation().str() == name)
			return true;
	}

	return false;
}

/* Is decl marked as exported?
 */
static bool is_exported(Decl *decl)
{
	return has_annotation(decl, "isl_export");
}

/* Collect all types and functions that are annotated "isl_export"
 * in "exported_types" and "exported_function".  Collect all function
 * declarations in "functions".
 *
 * We currently only consider single declarations.
 */
struct MyASTConsumer : public ASTConsumer {
	set<RecordDecl *> exported_types;
	set<FunctionDecl *> exported_functions;
	set<FunctionDecl *> functions;

	virtual HandleTopLevelDeclReturn HandleTopLevelDecl(DeclGroupRef D) {
		Decl *decl;

		if (!D.isSingleDecl())
			return HandleTopLevelDeclContinue;
		decl = D.getSingleDecl();
		if (isa<FunctionDecl>(decl))
			functions.insert(cast<FunctionDecl>(decl));
		if (!is_exported(decl))
			return HandleTopLevelDeclContinue;
		switch (decl->getKind()) {
		case Decl::Record:
			exported_types.insert(cast<RecordDecl>(decl));
			break;
		case Decl::Function:
			exported_functions.insert(cast<FunctionDecl>(decl));
			break;
		default:
			break;
		}
		return HandleTopLevelDeclContinue;
	}
};

#ifdef USE_ARRAYREF

#ifdef HAVE_CXXISPRODUCTION
static Driver *construct_driver(const char *binary, DiagnosticsEngine &Diags)
{
	return new Driver(binary, llvm::sys::getDefaultTargetTriple(),
			    "", false, false, Diags);
}
#elif defined(HAVE_ISPRODUCTION)
static Driver *construct_driver(const char *binary, DiagnosticsEngine &Diags)
{
	return new Driver(binary, llvm::sys::getDefaultTargetTriple(),
			    "", false, Diags);
}
#elif defined(DRIVER_CTOR_TAKES_DEFAULTIMAGENAME)
static Driver *construct_driver(const char *binary, DiagnosticsEngine &Diags)
{
	return new Driver(binary, llvm::sys::getDefaultTargetTriple(),
			    "", Diags);
}
#else
static Driver *construct_driver(const char *binary, DiagnosticsEngine &Diags)
{
	return new Driver(binary, llvm::sys::getDefaultTargetTriple(), Diags);
}
#endif

namespace clang { namespace driver { class Job; } }

/* Clang changed its API from 3.5 to 3.6 and once more in 3.7.
 * We fix this with a simple overloaded function here.
 */
struct ClangAPI {
	static Job *command(Job *J) { return J; }
	static Job *command(Job &J) { return &J; }
	static Command *command(Command &C) { return &C; }
};

/* Create a CompilerInvocation object that stores the command line
 * arguments constructed by the driver.
 * The arguments are mainly useful for setting up the system include
 * paths on newer clangs and on some platforms.
 */
static CompilerInvocation *construct_invocation(const char *filename,
	DiagnosticsEngine &Diags)
{
	const char *binary = CLANG_PREFIX"/bin/clang";
	const unique_ptr<Driver> driver(construct_driver(binary, Diags));
	std::vector<const char *> Argv;
	Argv.push_back(binary);
	Argv.push_back(filename);
	const unique_ptr<Compilation> compilation(
		driver->BuildCompilation(llvm::ArrayRef<const char *>(Argv)));
	JobList &Jobs = compilation->getJobs();

	Command *cmd = cast<Command>(ClangAPI::command(*Jobs.begin()));
	if (strcmp(cmd->getCreator().getName(), "clang"))
		return NULL;

	const ArgStringList *args = &cmd->getArguments();

	CompilerInvocation *invocation = new CompilerInvocation;
	CompilerInvocation::CreateFromArgs(*invocation, args->data() + 1,
						args->data() + args->size(),
						Diags);
	return invocation;
}

#else

static CompilerInvocation *construct_invocation(const char *filename,
	DiagnosticsEngine &Diags)
{
	return NULL;
}

#endif

#ifdef HAVE_BASIC_DIAGNOSTICOPTIONS_H

static TextDiagnosticPrinter *construct_printer(void)
{
	return new TextDiagnosticPrinter(llvm::errs(), new DiagnosticOptions());
}

#else

static TextDiagnosticPrinter *construct_printer(void)
{
	DiagnosticOptions DO;
	return new TextDiagnosticPrinter(llvm::errs(), DO);
}

#endif

#ifdef CREATETARGETINFO_TAKES_SHARED_PTR

static TargetInfo *create_target_info(CompilerInstance *Clang,
	DiagnosticsEngine &Diags)
{
	shared_ptr<TargetOptions> TO = Clang->getInvocation().TargetOpts;
	TO->Triple = llvm::sys::getDefaultTargetTriple();
	return TargetInfo::CreateTargetInfo(Diags, TO);
}

#elif defined(CREATETARGETINFO_TAKES_POINTER)

static TargetInfo *create_target_info(CompilerInstance *Clang,
	DiagnosticsEngine &Diags)
{
	TargetOptions &TO = Clang->getTargetOpts();
	TO.Triple = llvm::sys::getDefaultTargetTriple();
	return TargetInfo::CreateTargetInfo(Diags, &TO);
}

#else

static TargetInfo *create_target_info(CompilerInstance *Clang,
	DiagnosticsEngine &Diags)
{
	TargetOptions &TO = Clang->getTargetOpts();
	TO.Triple = llvm::sys::getDefaultTargetTriple();
	return TargetInfo::CreateTargetInfo(Diags, TO);
}

#endif

#ifdef CREATEDIAGNOSTICS_TAKES_ARG

static void create_diagnostics(CompilerInstance *Clang)
{
	Clang->createDiagnostics(0, NULL, construct_printer());
}

#else

static void create_diagnostics(CompilerInstance *Clang)
{
	Clang->createDiagnostics(construct_printer());
}

#endif

#ifdef CREATEPREPROCESSOR_TAKES_TUKIND

static void create_preprocessor(CompilerInstance *Clang)
{
	Clang->createPreprocessor(TU_Complete);
}

#else

static void create_preprocessor(CompilerInstance *Clang)
{
	Clang->createPreprocessor();
}

#endif

#ifdef ADDPATH_TAKES_4_ARGUMENTS

void add_path(HeaderSearchOptions &HSO, string Path)
{
	HSO.AddPath(Path, frontend::Angled, false, false);
}

#else

void add_path(HeaderSearchOptions &HSO, string Path)
{
	HSO.AddPath(Path, frontend::Angled, true, false, false);
}

#endif

#ifdef HAVE_SETMAINFILEID

static void create_main_file_id(SourceManager &SM, const FileEntry *file)
{
	SM.setMainFileID(SM.createFileID(file, SourceLocation(),
					SrcMgr::C_User));
}

#else

static void create_main_file_id(SourceManager &SM, const FileEntry *file)
{
	SM.createMainFileID(file);
}

#endif

#ifdef SETLANGDEFAULTS_TAKES_5_ARGUMENTS

static void set_lang_defaults(CompilerInstance *Clang)
{
	PreprocessorOptions &PO = Clang->getPreprocessorOpts();
	TargetOptions &TO = Clang->getTargetOpts();
	llvm::Triple T(TO.Triple);
	CompilerInvocation::setLangDefaults(Clang->getLangOpts(), IK_C, T, PO,
					    LangStandard::lang_unspecified);
}

#else

static void set_lang_defaults(CompilerInstance *Clang)
{
	CompilerInvocation::setLangDefaults(Clang->getLangOpts(), IK_C,
					    LangStandard::lang_unspecified);
}

#endif

int main(int argc, char *argv[])
{
	llvm::cl::ParseCommandLineOptions(argc, argv);

	CompilerInstance *Clang = new CompilerInstance();
	create_diagnostics(Clang);
	DiagnosticsEngine &Diags = Clang->getDiagnostics();
	Diags.setSuppressSystemWarnings(true);
	CompilerInvocation *invocation =
		construct_invocation(InputFilename.c_str(), Diags);
	if (invocation)
		Clang->setInvocation(invocation);
	Clang->createFileManager();
	Clang->createSourceManager(Clang->getFileManager());
	TargetInfo *target = create_target_info(Clang, Diags);
	Clang->setTarget(target);
	set_lang_defaults(Clang);
	HeaderSearchOptions &HSO = Clang->getHeaderSearchOpts();
	LangOptions &LO = Clang->getLangOpts();
	PreprocessorOptions &PO = Clang->getPreprocessorOpts();
	HSO.ResourceDir = ResourceDir;

	for (llvm::cl::list<string>::size_type i = 0; i < Includes.size(); ++i)
		add_path(HSO, Includes[i]);

	PO.addMacroDef("__isl_give=__attribute__((annotate(\"isl_give\")))");
	PO.addMacroDef("__isl_keep=__attribute__((annotate(\"isl_keep\")))");
	PO.addMacroDef("__isl_take=__attribute__((annotate(\"isl_take\")))");
	PO.addMacroDef("__isl_export=__attribute__((annotate(\"isl_export\")))");
	PO.addMacroDef("__isl_overload="
	    "__attribute__((annotate(\"isl_overload\"))) "
	    "__attribute__((annotate(\"isl_export\")))");
	PO.addMacroDef("__isl_constructor=__attribute__((annotate(\"isl_constructor\"))) __attribute__((annotate(\"isl_export\")))");
	PO.addMacroDef("__isl_subclass(super)=__attribute__((annotate(\"isl_subclass(\" #super \")\"))) __attribute__((annotate(\"isl_export\")))");

	create_preprocessor(Clang);
	Preprocessor &PP = Clang->getPreprocessor();

	PP.getBuiltinInfo().initializeBuiltins(PP.getIdentifierTable(), LO);

	const FileEntry *file = Clang->getFileManager().getFile(InputFilename);
	assert(file);
	create_main_file_id(Clang->getSourceManager(), file);

	Clang->createASTContext();
	MyASTConsumer consumer;
	Sema *sema = new Sema(PP, Clang->getASTContext(), consumer);

	Diags.getClient()->BeginSourceFile(LO, &PP);
	ParseAST(*sema);
	Diags.getClient()->EndSourceFile();

	generate_python(consumer.exported_types, consumer.exported_functions,
			consumer.functions);

	delete sema;
	delete Clang;
	llvm::llvm_shutdown();

	return 0;
}
