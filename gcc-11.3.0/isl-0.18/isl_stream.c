/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 */

#include <ctype.h>
#include <string.h>
#include <isl/ctx.h>
#include <isl_stream_private.h>
#include <isl/map.h>
#include <isl/aff.h>
#include <isl_val_private.h>

struct isl_keyword {
	char			*name;
	enum isl_token_type	type;
};

static int same_name(const void *entry, const void *val)
{
	const struct isl_keyword *keyword = (const struct isl_keyword *)entry;

	return !strcmp(keyword->name, val);
}

enum isl_token_type isl_stream_register_keyword(__isl_keep isl_stream *s,
	const char *name)
{
	struct isl_hash_table_entry *entry;
	struct isl_keyword *keyword;
	uint32_t name_hash;

	if (!s->keywords) {
		s->keywords = isl_hash_table_alloc(s->ctx, 10);
		if (!s->keywords)
			return ISL_TOKEN_ERROR;
		s->next_type = ISL_TOKEN_LAST;
	}

	name_hash = isl_hash_string(isl_hash_init(), name);

	entry = isl_hash_table_find(s->ctx, s->keywords, name_hash,
					same_name, name, 1);
	if (!entry)
		return ISL_TOKEN_ERROR;
	if (entry->data) {
		keyword = entry->data;
		return keyword->type;
	}

	keyword = isl_calloc_type(s->ctx, struct isl_keyword);
	if (!keyword)
		return ISL_TOKEN_ERROR;
	keyword->type = s->next_type++;
	keyword->name = strdup(name);
	if (!keyword->name) {
		free(keyword);
		return ISL_TOKEN_ERROR;
	}
	entry->data = keyword;

	return keyword->type;
}

struct isl_token *isl_token_new(isl_ctx *ctx,
	int line, int col, unsigned on_new_line)
{
	struct isl_token *tok = isl_alloc_type(ctx, struct isl_token);
	if (!tok)
		return NULL;
	tok->line = line;
	tok->col = col;
	tok->on_new_line = on_new_line;
	tok->is_keyword = 0;
	tok->u.s = NULL;
	return tok;
}

/* Return the type of "tok".
 */
int isl_token_get_type(struct isl_token *tok)
{
	return tok ? tok->type : ISL_TOKEN_ERROR;
}

/* Given a token of type ISL_TOKEN_VALUE, return the value it represents.
 */
__isl_give isl_val *isl_token_get_val(isl_ctx *ctx, struct isl_token *tok)
{
	if (!tok)
		return NULL;
	if (tok->type != ISL_TOKEN_VALUE)
		isl_die(ctx, isl_error_invalid, "not a value token",
			return NULL);

	return isl_val_int_from_isl_int(ctx, tok->u.v);
}

/* Given a token with a string representation, return a copy of this string.
 */
__isl_give char *isl_token_get_str(isl_ctx *ctx, struct isl_token *tok)
{
	if (!tok)
		return NULL;
	if (!tok->u.s)
		isl_die(ctx, isl_error_invalid,
			"token does not have a string representation",
			return NULL);

	return strdup(tok->u.s);
}

void isl_token_free(struct isl_token *tok)
{
	if (!tok)
		return;
	if (tok->type == ISL_TOKEN_VALUE)
		isl_int_clear(tok->u.v);
	else if (tok->type == ISL_TOKEN_MAP)
		isl_map_free(tok->u.map);
	else if (tok->type == ISL_TOKEN_AFF)
		isl_pw_aff_free(tok->u.pwaff);
	else
		free(tok->u.s);
	free(tok);
}

void isl_stream_error(__isl_keep isl_stream *s, struct isl_token *tok,
	char *msg)
{
	int line = tok ? tok->line : s->line;
	int col = tok ? tok->col : s->col;
	fprintf(stderr, "syntax error (%d, %d): %s\n", line, col, msg);
	if (tok) {
		if (tok->type < 256)
			fprintf(stderr, "got '%c'\n", tok->type);
		else if (tok->type == ISL_TOKEN_IDENT)
			fprintf(stderr, "got ident '%s'\n", tok->u.s);
		else if (tok->is_keyword)
			fprintf(stderr, "got keyword '%s'\n", tok->u.s);
		else if (tok->type == ISL_TOKEN_VALUE) {
			fprintf(stderr, "got value '");
			isl_int_print(stderr, tok->u.v, 0);
			fprintf(stderr, "'\n");
		} else if (tok->type == ISL_TOKEN_MAP) {
			isl_printer *p;
			fprintf(stderr, "got map '");
			p = isl_printer_to_file(s->ctx, stderr);
			p = isl_printer_print_map(p, tok->u.map);
			isl_printer_free(p);
			fprintf(stderr, "'\n");
		} else if (tok->type == ISL_TOKEN_AFF) {
			isl_printer *p;
			fprintf(stderr, "got affine expression '");
			p = isl_printer_to_file(s->ctx, stderr);
			p = isl_printer_print_pw_aff(p, tok->u.pwaff);
			isl_printer_free(p);
			fprintf(stderr, "'\n");
		} else if (tok->u.s)
			fprintf(stderr, "got token '%s'\n", tok->u.s);
		else
			fprintf(stderr, "got token type %d\n", tok->type);
	}
}

static __isl_give isl_stream* isl_stream_new(struct isl_ctx *ctx)
{
	int i;
	isl_stream *s = isl_calloc_type(ctx, struct isl_stream);
	if (!s)
		return NULL;
	s->ctx = ctx;
	isl_ctx_ref(s->ctx);
	s->file = NULL;
	s->str = NULL;
	s->len = 0;
	s->line = 1;
	s->col = 1;
	s->eof = 0;
	s->last_line = 0;
	s->c = -1;
	s->n_un = 0;
	for (i = 0; i < 5; ++i)
		s->tokens[i] = NULL;
	s->n_token = 0;
	s->keywords = NULL;
	s->size = 256;
	s->buffer = isl_alloc_array(ctx, char, s->size);
	if (!s->buffer)
		goto error;
	return s;
error:
	isl_stream_free(s);
	return NULL;
}

__isl_give isl_stream* isl_stream_new_file(struct isl_ctx *ctx, FILE *file)
{
	isl_stream *s = isl_stream_new(ctx);
	if (!s)
		return NULL;
	s->file = file;
	return s;
}

__isl_give isl_stream* isl_stream_new_str(struct isl_ctx *ctx, const char *str)
{
	isl_stream *s;
	if (!str)
		return NULL;
	s = isl_stream_new(ctx);
	if (!s)
		return NULL;
	s->str = str;
	return s;
}

/* Read a character from the stream and advance s->line and s->col
 * to point to the next character.
 */
static int stream_getc(__isl_keep isl_stream *s)
{
	int c;
	if (s->eof)
		return -1;
	if (s->n_un)
		return s->c = s->un[--s->n_un];
	if (s->file)
		c = fgetc(s->file);
	else {
		c = *s->str++;
		if (c == '\0')
			c = -1;
	}
	if (c == -1)
		s->eof = 1;
	else if (c == '\n') {
		s->line++;
		s->col = 1;
	} else
		s->col++;
	s->c = c;
	return c;
}

static void isl_stream_ungetc(__isl_keep isl_stream *s, int c)
{
	isl_assert(s->ctx, s->n_un < 5, return);
	s->un[s->n_un++] = c;
	s->c = -1;
}

/* Read a character from the stream, skipping pairs of '\\' and '\n'.
 * Set s->start_line and s->start_col to the line and column
 * of the returned character.
 */
static int isl_stream_getc(__isl_keep isl_stream *s)
{
	int c;

	do {
		s->start_line = s->line;
		s->start_col = s->col;
		c = stream_getc(s);
		if (c != '\\')
			return c;
		c = stream_getc(s);
	} while (c == '\n');

	isl_stream_ungetc(s, c);

	return '\\';
}

static int isl_stream_push_char(__isl_keep isl_stream *s, int c)
{
	if (s->len >= s->size) {
		char *buffer;
		s->size = (3*s->size)/2;
		buffer = isl_realloc_array(s->ctx, s->buffer, char, s->size);
		if (!buffer)
			return -1;
		s->buffer = buffer;
	}
	s->buffer[s->len++] = c;
	return 0;
}

void isl_stream_push_token(__isl_keep isl_stream *s, struct isl_token *tok)
{
	isl_assert(s->ctx, s->n_token < 5, return);
	s->tokens[s->n_token++] = tok;
}

static enum isl_token_type check_keywords(__isl_keep isl_stream *s)
{
	struct isl_hash_table_entry *entry;
	struct isl_keyword *keyword;
	uint32_t name_hash;

	if (!strcasecmp(s->buffer, "exists"))
		return ISL_TOKEN_EXISTS;
	if (!strcasecmp(s->buffer, "and"))
		return ISL_TOKEN_AND;
	if (!strcasecmp(s->buffer, "or"))
		return ISL_TOKEN_OR;
	if (!strcasecmp(s->buffer, "implies"))
		return ISL_TOKEN_IMPLIES;
	if (!strcasecmp(s->buffer, "not"))
		return ISL_TOKEN_NOT;
	if (!strcasecmp(s->buffer, "infty"))
		return ISL_TOKEN_INFTY;
	if (!strcasecmp(s->buffer, "infinity"))
		return ISL_TOKEN_INFTY;
	if (!strcasecmp(s->buffer, "NaN"))
		return ISL_TOKEN_NAN;
	if (!strcasecmp(s->buffer, "min"))
		return ISL_TOKEN_MIN;
	if (!strcasecmp(s->buffer, "max"))
		return ISL_TOKEN_MAX;
	if (!strcasecmp(s->buffer, "rat"))
		return ISL_TOKEN_RAT;
	if (!strcasecmp(s->buffer, "true"))
		return ISL_TOKEN_TRUE;
	if (!strcasecmp(s->buffer, "false"))
		return ISL_TOKEN_FALSE;
	if (!strcasecmp(s->buffer, "ceild"))
		return ISL_TOKEN_CEILD;
	if (!strcasecmp(s->buffer, "floord"))
		return ISL_TOKEN_FLOORD;
	if (!strcasecmp(s->buffer, "mod"))
		return ISL_TOKEN_MOD;
	if (!strcasecmp(s->buffer, "ceil"))
		return ISL_TOKEN_CEIL;
	if (!strcasecmp(s->buffer, "floor"))
		return ISL_TOKEN_FLOOR;

	if (!s->keywords)
		return ISL_TOKEN_IDENT;

	name_hash = isl_hash_string(isl_hash_init(), s->buffer);
	entry = isl_hash_table_find(s->ctx, s->keywords, name_hash, same_name,
					s->buffer, 0);
	if (entry) {
		keyword = entry->data;
		return keyword->type;
	}

	return ISL_TOKEN_IDENT;
}

int isl_stream_skip_line(__isl_keep isl_stream *s)
{
	int c;

	while ((c = isl_stream_getc(s)) != -1 && c != '\n')
		/* nothing */
		;

	return c == -1 ? -1 : 0;
}

static struct isl_token *next_token(__isl_keep isl_stream *s, int same_line)
{
	int c;
	struct isl_token *tok = NULL;
	int line, col;
	int old_line = s->last_line;

	if (s->n_token) {
		if (same_line && s->tokens[s->n_token - 1]->on_new_line)
			return NULL;
		return s->tokens[--s->n_token];
	}

	if (same_line && s->c == '\n')
		return NULL;

	s->len = 0;

	/* skip spaces and comment lines */
	while ((c = isl_stream_getc(s)) != -1) {
		if (c == '#') {
			if (isl_stream_skip_line(s) < 0)
				break;
			c = '\n';
			if (same_line)
				break;
		} else if (!isspace(c) || (same_line && c == '\n'))
			break;
	}

	line = s->start_line;
	col = s->start_col;

	if (c == -1 || (same_line && c == '\n'))
		return NULL;
	s->last_line = line;

	if (c == '(' ||
	    c == ')' ||
	    c == '+' ||
	    c == '*' ||
	    c == '%' ||
	    c == '?' ||
	    c == '^' ||
	    c == '@' ||
	    c == '$' ||
	    c == ',' ||
	    c == '.' ||
	    c == ';' ||
	    c == '[' ||
	    c == ']' ||
	    c == '{' ||
	    c == '}') {
		tok = isl_token_new(s->ctx, line, col, old_line != line);
		if (!tok)
			return NULL;
		tok->type = (enum isl_token_type)c;
		return tok;
	}
	if (c == '-') {
		int c;
		if ((c = isl_stream_getc(s)) == '>') {
			tok = isl_token_new(s->ctx, line, col, old_line != line);
			if (!tok)
				return NULL;
			tok->u.s = strdup("->");
			tok->type = ISL_TOKEN_TO;
			return tok;
		}
		if (c != -1)
			isl_stream_ungetc(s, c);
		if (!isdigit(c)) {
			tok = isl_token_new(s->ctx, line, col, old_line != line);
			if (!tok)
				return NULL;
			tok->type = (enum isl_token_type) '-';
			return tok;
		}
	}
	if (c == '-' || isdigit(c)) {
		int minus = c == '-';
		tok = isl_token_new(s->ctx, line, col, old_line != line);
		if (!tok)
			return NULL;
		tok->type = ISL_TOKEN_VALUE;
		isl_int_init(tok->u.v);
		if (isl_stream_push_char(s, c))
			goto error;
		while ((c = isl_stream_getc(s)) != -1 && isdigit(c))
			if (isl_stream_push_char(s, c))
				goto error;
		if (c != -1)
			isl_stream_ungetc(s, c);
		isl_stream_push_char(s, '\0');
		isl_int_read(tok->u.v, s->buffer);
		if (minus && isl_int_is_zero(tok->u.v)) {
			tok->col++;
			tok->on_new_line = 0;
			isl_stream_push_token(s, tok);
			tok = isl_token_new(s->ctx, line, col, old_line != line);
			if (!tok)
				return NULL;
			tok->type = (enum isl_token_type) '-';
		}
		return tok;
	}
	if (isalpha(c) || c == '_') {
		tok = isl_token_new(s->ctx, line, col, old_line != line);
		if (!tok)
			return NULL;
		isl_stream_push_char(s, c);
		while ((c = isl_stream_getc(s)) != -1 &&
				(isalnum(c) || c == '_'))
			isl_stream_push_char(s, c);
		if (c != -1)
			isl_stream_ungetc(s, c);
		while ((c = isl_stream_getc(s)) != -1 && c == '\'')
			isl_stream_push_char(s, c);
		if (c != -1)
			isl_stream_ungetc(s, c);
		isl_stream_push_char(s, '\0');
		tok->type = check_keywords(s);
		if (tok->type != ISL_TOKEN_IDENT)
			tok->is_keyword = 1;
		tok->u.s = strdup(s->buffer);
		if (!tok->u.s)
			goto error;
		return tok;
	}
	if (c == '"') {
		tok = isl_token_new(s->ctx, line, col, old_line != line);
		if (!tok)
			return NULL;
		tok->type = ISL_TOKEN_STRING;
		tok->u.s = NULL;
		while ((c = isl_stream_getc(s)) != -1 && c != '"' && c != '\n')
			isl_stream_push_char(s, c);
		if (c != '"') {
			isl_stream_error(s, NULL, "unterminated string");
			goto error;
		}
		isl_stream_push_char(s, '\0');
		tok->u.s = strdup(s->buffer);
		return tok;
	}
	if (c == '=') {
		int c;
		tok = isl_token_new(s->ctx, line, col, old_line != line);
		if (!tok)
			return NULL;
		if ((c = isl_stream_getc(s)) == '=') {
			tok->u.s = strdup("==");
			tok->type = ISL_TOKEN_EQ_EQ;
			return tok;
		}
		if (c != -1)
			isl_stream_ungetc(s, c);
		tok->type = (enum isl_token_type) '=';
		return tok;
	}
	if (c == ':') {
		int c;
		tok = isl_token_new(s->ctx, line, col, old_line != line);
		if (!tok)
			return NULL;
		if ((c = isl_stream_getc(s)) == '=') {
			tok->u.s = strdup(":=");
			tok->type = ISL_TOKEN_DEF;
			return tok;
		}
		if (c != -1)
			isl_stream_ungetc(s, c);
		tok->type = (enum isl_token_type) ':';
		return tok;
	}
	if (c == '>') {
		int c;
		tok = isl_token_new(s->ctx, line, col, old_line != line);
		if (!tok)
			return NULL;
		if ((c = isl_stream_getc(s)) == '=') {
			tok->u.s = strdup(">=");
			tok->type = ISL_TOKEN_GE;
			return tok;
		} else if (c == '>') {
			if ((c = isl_stream_getc(s)) == '=') {
				tok->u.s = strdup(">>=");
				tok->type = ISL_TOKEN_LEX_GE;
				return tok;
			}
			tok->u.s = strdup(">>");
			tok->type = ISL_TOKEN_LEX_GT;
		} else {
			tok->u.s = strdup(">");
			tok->type = ISL_TOKEN_GT;
		}
		if (c != -1)
			isl_stream_ungetc(s, c);
		return tok;
	}
	if (c == '<') {
		int c;
		tok = isl_token_new(s->ctx, line, col, old_line != line);
		if (!tok)
			return NULL;
		if ((c = isl_stream_getc(s)) == '=') {
			tok->u.s = strdup("<=");
			tok->type = ISL_TOKEN_LE;
			return tok;
		} else if (c == '<') {
			if ((c = isl_stream_getc(s)) == '=') {
				tok->u.s = strdup("<<=");
				tok->type = ISL_TOKEN_LEX_LE;
				return tok;
			}
			tok->u.s = strdup("<<");
			tok->type = ISL_TOKEN_LEX_LT;
		} else {
			tok->u.s = strdup("<");
			tok->type = ISL_TOKEN_LT;
		}
		if (c != -1)
			isl_stream_ungetc(s, c);
		return tok;
	}
	if (c == '&') {
		tok = isl_token_new(s->ctx, line, col, old_line != line);
		if (!tok)
			return NULL;
		tok->type = ISL_TOKEN_AND;
		if ((c = isl_stream_getc(s)) != '&' && c != -1) {
			tok->u.s = strdup("&");
			isl_stream_ungetc(s, c);
		} else
			tok->u.s = strdup("&&");
		return tok;
	}
	if (c == '|') {
		tok = isl_token_new(s->ctx, line, col, old_line != line);
		if (!tok)
			return NULL;
		tok->type = ISL_TOKEN_OR;
		if ((c = isl_stream_getc(s)) != '|' && c != -1) {
			tok->u.s = strdup("|");
			isl_stream_ungetc(s, c);
		} else
			tok->u.s = strdup("||");
		return tok;
	}
	if (c == '/') {
		tok = isl_token_new(s->ctx, line, col, old_line != line);
		if (!tok)
			return NULL;
		if ((c = isl_stream_getc(s)) != '\\' && c != -1) {
			tok->type = (enum isl_token_type) '/';
			isl_stream_ungetc(s, c);
		} else {
			tok->u.s = strdup("/\\");
			tok->type = ISL_TOKEN_AND;
		}
		return tok;
	}
	if (c == '\\') {
		tok = isl_token_new(s->ctx, line, col, old_line != line);
		if (!tok)
			return NULL;
		if ((c = isl_stream_getc(s)) != '/' && c != -1) {
			tok->type = (enum isl_token_type) '\\';
			isl_stream_ungetc(s, c);
		} else {
			tok->u.s = strdup("\\/");
			tok->type = ISL_TOKEN_OR;
		}
		return tok;
	}
	if (c == '!') {
		tok = isl_token_new(s->ctx, line, col, old_line != line);
		if (!tok)
			return NULL;
		if ((c = isl_stream_getc(s)) == '=') {
			tok->u.s = strdup("!=");
			tok->type = ISL_TOKEN_NE;
			return tok;
		} else {
			tok->type = ISL_TOKEN_NOT;
			tok->u.s = strdup("!");
		}
		if (c != -1)
			isl_stream_ungetc(s, c);
		return tok;
	}

	tok = isl_token_new(s->ctx, line, col, old_line != line);
	if (!tok)
		return NULL;
	tok->type = ISL_TOKEN_UNKNOWN;
	return tok;
error:
	isl_token_free(tok);
	return NULL;
}

struct isl_token *isl_stream_next_token(__isl_keep isl_stream *s)
{
	return next_token(s, 0);
}

struct isl_token *isl_stream_next_token_on_same_line(__isl_keep isl_stream *s)
{
	return next_token(s, 1);
}

int isl_stream_eat_if_available(__isl_keep isl_stream *s, int type)
{
	struct isl_token *tok;

	tok = isl_stream_next_token(s);
	if (!tok)
		return 0;
	if (tok->type == type) {
		isl_token_free(tok);
		return 1;
	}
	isl_stream_push_token(s, tok);
	return 0;
}

int isl_stream_next_token_is(__isl_keep isl_stream *s, int type)
{
	struct isl_token *tok;
	int r;

	tok = isl_stream_next_token(s);
	if (!tok)
		return 0;
	r = tok->type == type;
	isl_stream_push_token(s, tok);
	return r;
}

char *isl_stream_read_ident_if_available(__isl_keep isl_stream *s)
{
	struct isl_token *tok;

	tok = isl_stream_next_token(s);
	if (!tok)
		return NULL;
	if (tok->type == ISL_TOKEN_IDENT) {
		char *ident = strdup(tok->u.s);
		isl_token_free(tok);
		return ident;
	}
	isl_stream_push_token(s, tok);
	return NULL;
}

int isl_stream_eat(__isl_keep isl_stream *s, int type)
{
	struct isl_token *tok;

	tok = isl_stream_next_token(s);
	if (!tok)
		return -1;
	if (tok->type == type) {
		isl_token_free(tok);
		return 0;
	}
	isl_stream_error(s, tok, "expecting other token");
	isl_stream_push_token(s, tok);
	return -1;
}

int isl_stream_is_empty(__isl_keep isl_stream *s)
{
	struct isl_token *tok;

	tok = isl_stream_next_token(s);

	if (!tok)
		return 1;

	isl_stream_push_token(s, tok);
	return 0;
}

static isl_stat free_keyword(void **p, void *user)
{
	struct isl_keyword *keyword = *p;

	free(keyword->name);
	free(keyword);

	return isl_stat_ok;
}

void isl_stream_flush_tokens(__isl_keep isl_stream *s)
{
	int i;

	if (!s)
		return;
	for (i = 0; i < s->n_token; ++i)
		isl_token_free(s->tokens[i]);
	s->n_token = 0;
}

isl_ctx *isl_stream_get_ctx(__isl_keep isl_stream *s)
{
	return s ? s->ctx : NULL;
}

void isl_stream_free(__isl_take isl_stream *s)
{
	if (!s)
		return;
	free(s->buffer);
	if (s->n_token != 0) {
		struct isl_token *tok = isl_stream_next_token(s);
		isl_stream_error(s, tok, "unexpected token");
		isl_token_free(tok);
	}
	if (s->keywords) {
		isl_hash_table_foreach(s->ctx, s->keywords, &free_keyword, NULL);
		isl_hash_table_free(s->ctx, s->keywords);
	}
	free(s->yaml_state);
	free(s->yaml_indent);
	isl_ctx_deref(s->ctx);
	free(s);
}

/* Push "state" onto the stack of currently active YAML elements.
 * The caller is responsible for setting the corresponding indentation.
 * Return 0 on success and -1 on failure.
 */
static int push_state(__isl_keep isl_stream *s, enum isl_yaml_state state)
{
	if (s->yaml_size < s->yaml_depth + 1) {
		int *indent;
		enum isl_yaml_state *state;

		state = isl_realloc_array(s->ctx, s->yaml_state,
					enum isl_yaml_state, s->yaml_depth + 1);
		if (!state)
			return -1;
		s->yaml_state = state;

		indent = isl_realloc_array(s->ctx, s->yaml_indent,
					int, s->yaml_depth + 1);
		if (!indent)
			return -1;
		s->yaml_indent = indent;

		s->yaml_size = s->yaml_depth + 1;
	}

	s->yaml_state[s->yaml_depth] = state;
	s->yaml_depth++;

	return 0;
}

/* Remove the innermost active YAML element from the stack.
 * Return 0 on success and -1 on failure.
 */
static int pop_state(__isl_keep isl_stream *s)
{
	if (!s)
		return -1;
	if (s->yaml_depth < 1)
		isl_die(isl_stream_get_ctx(s), isl_error_invalid,
			"not in YAML construct", return -1);

	s->yaml_depth--;

	return 0;
}

/* Set the state of the innermost active YAML element to "state".
 * Return 0 on success and -1 on failure.
 */
static int update_state(__isl_keep isl_stream *s, enum isl_yaml_state state)
{
	if (!s)
		return -1;
	if (s->yaml_depth < 1)
		isl_die(isl_stream_get_ctx(s), isl_error_invalid,
			"not in YAML construct", return -1);

	s->yaml_state[s->yaml_depth - 1] = state;

	return 0;
}

/* Return the state of the innermost active YAML element.
 * Return isl_yaml_none if we are not inside any YAML element.
 */
static enum isl_yaml_state current_state(__isl_keep isl_stream *s)
{
	if (!s)
		return isl_yaml_none;
	if (s->yaml_depth < 1)
		return isl_yaml_none;
	return s->yaml_state[s->yaml_depth - 1];
}

/* Set the indentation of the innermost active YAML element to "indent".
 * If "indent" is equal to ISL_YAML_INDENT_FLOW, then this means
 * that the current elemient is in flow format.
 */
static int set_yaml_indent(__isl_keep isl_stream *s, int indent)
{
	if (s->yaml_depth < 1)
		isl_die(s->ctx, isl_error_internal,
			"not in YAML element", return -1);

	s->yaml_indent[s->yaml_depth - 1] = indent;

	return 0;
}

/* Return the indentation of the innermost active YAML element
 * of -1 on error.
 */
static int get_yaml_indent(__isl_keep isl_stream *s)
{
	if (s->yaml_depth < 1)
		isl_die(s->ctx, isl_error_internal,
			"not in YAML element", return -1);

	return s->yaml_indent[s->yaml_depth - 1];
}

/* Move to the next state at the innermost level.
 * Return 1 if successful.
 * Return 0 if we are at the end of the innermost level.
 * Return -1 on error.
 *
 * If we are in state isl_yaml_mapping_key_start, then we have just
 * started a mapping and we are expecting a key.  If the mapping started
 * with a '{', then we check if the next token is a '}'.  If so,
 * then the mapping is empty and there is no next state at this level.
 * Otherwise, we assume that there is at least one key (the one from
 * which we derived the indentation in isl_stream_yaml_read_start_mapping.
 *
 * If we are in state isl_yaml_mapping_key, then the we expect a colon
 * followed by a value, so there is always a next state unless
 * some error occurs.
 *
 * If we are in state isl_yaml_mapping_val, then there may or may
 * not be a subsequent key in the same mapping.
 * In flow format, the next key is preceded by a comma.
 * In block format, the next key has the same indentation as the first key.
 * If the first token has a smaller indentation, then we have reached
 * the end of the current mapping.
 *
 * If we are in state isl_yaml_sequence_start, then we have just
 * started a sequence.  If the sequence started with a '[',
 * then we check if the next token is a ']'.  If so, then the sequence
 * is empty and there is no next state at this level.
 * Otherwise, we assume that there is at least one element in the sequence
 * (the one from which we derived the indentation in
 * isl_stream_yaml_read_start_sequence.
 *
 * If we are in state isl_yaml_sequence, then there may or may
 * not be a subsequent element in the same sequence.
 * In flow format, the next element is preceded by a comma.
 * In block format, the next element is introduced by a dash with
 * the same indentation as that of the first element.
 * If the first token is not a dash or if it has a smaller indentation,
 * then we have reached the end of the current sequence.
 */
int isl_stream_yaml_next(__isl_keep isl_stream *s)
{
	struct isl_token *tok;
	enum isl_yaml_state state;
	int indent;

	state = current_state(s);
	if (state == isl_yaml_none)
		isl_die(s->ctx, isl_error_invalid,
			"not in YAML element", return -1);
	switch (state) {
	case isl_yaml_mapping_key_start:
		if (get_yaml_indent(s) == ISL_YAML_INDENT_FLOW &&
		    isl_stream_next_token_is(s, '}'))
			return 0;
		if (update_state(s, isl_yaml_mapping_key) < 0)
			return -1;
		return 1;
	case isl_yaml_mapping_key:
		tok = isl_stream_next_token(s);
		if (!tok) {
			if (s->eof)
				isl_stream_error(s, NULL, "unexpected EOF");
			return -1;
		}
		if (tok->type == ':') {
			isl_token_free(tok);
			if (update_state(s, isl_yaml_mapping_val) < 0)
				return -1;
			return 1;
		}
		isl_stream_error(s, tok, "expecting ':'");
		isl_stream_push_token(s, tok);
		return -1;
	case isl_yaml_mapping_val:
		if (get_yaml_indent(s) == ISL_YAML_INDENT_FLOW) {
			if (!isl_stream_eat_if_available(s, ','))
				return 0;
			if (update_state(s, isl_yaml_mapping_key) < 0)
				return -1;
			return 1;
		}
		tok = isl_stream_next_token(s);
		if (!tok)
			return 0;
		indent = tok->col - 1;
		isl_stream_push_token(s, tok);
		if (indent < get_yaml_indent(s))
			return 0;
		if (update_state(s, isl_yaml_mapping_key) < 0)
			return -1;
		return 1;
	case isl_yaml_sequence_start:
		if (get_yaml_indent(s) == ISL_YAML_INDENT_FLOW) {
			if (isl_stream_next_token_is(s, ']'))
				return 0;
			if (update_state(s, isl_yaml_sequence) < 0)
				return -1;
			return 1;
		}
		tok = isl_stream_next_token(s);
		if (!tok) {
			if (s->eof)
				isl_stream_error(s, NULL, "unexpected EOF");
			return -1;
		}
		if (tok->type == '-') {
			isl_token_free(tok);
			if (update_state(s, isl_yaml_sequence) < 0)
				return -1;
			return 1;
		}
		isl_stream_error(s, tok, "expecting '-'");
		isl_stream_push_token(s, tok);
		return 0;
	case isl_yaml_sequence:
		if (get_yaml_indent(s) == ISL_YAML_INDENT_FLOW)
			return isl_stream_eat_if_available(s, ',');
		tok = isl_stream_next_token(s);
		if (!tok)
			return 0;
		indent = tok->col - 1;
		if (indent < get_yaml_indent(s) || tok->type != '-') {
			isl_stream_push_token(s, tok);
			return 0;
		}
		isl_token_free(tok);
		return 1;
	default:
		isl_die(s->ctx, isl_error_internal,
			"unexpected state", return 0);
	}
}

/* Start reading a YAML mapping.
 * Return 0 on success and -1 on error.
 *
 * If the first token on the stream is a '{' then we remove this token
 * from the stream and keep track of the fact that the mapping
 * is given in flow format.
 * Otherwise, we assume the first token is the first key of the mapping and
 * keep track of its indentation, but keep the token on the stream.
 * In both cases, the next token we expect is the first key of the mapping.
 */
int isl_stream_yaml_read_start_mapping(__isl_keep isl_stream *s)
{
	struct isl_token *tok;
	int indent;

	if (push_state(s, isl_yaml_mapping_key_start) < 0)
		return -1;

	tok = isl_stream_next_token(s);
	if (!tok) {
		if (s->eof)
			isl_stream_error(s, NULL, "unexpected EOF");
		return -1;
	}
	if (isl_token_get_type(tok) == '{') {
		isl_token_free(tok);
		return set_yaml_indent(s, ISL_YAML_INDENT_FLOW);
	}
	indent = tok->col - 1;
	isl_stream_push_token(s, tok);

	return set_yaml_indent(s, indent);
}

/* Finish reading a YAML mapping.
 * Return 0 on success and -1 on error.
 *
 * If the mapping started with a '{', then we expect a '}' to close
 * the mapping.
 * Otherwise, we double-check that the next token (if any)
 * has a smaller indentation than that of the current mapping.
 */
int isl_stream_yaml_read_end_mapping(__isl_keep isl_stream *s)
{
	struct isl_token *tok;
	int indent;

	if (get_yaml_indent(s) == ISL_YAML_INDENT_FLOW) {
		if (isl_stream_eat(s, '}') < 0)
			return -1;
		return pop_state(s);
	}

	tok = isl_stream_next_token(s);
	if (!tok)
		return pop_state(s);

	indent = tok->col - 1;
	isl_stream_push_token(s, tok);

	if (indent >= get_yaml_indent(s))
		isl_die(isl_stream_get_ctx(s), isl_error_invalid,
			"mapping not finished", return -1);

	return pop_state(s);
}

/* Start reading a YAML sequence.
 * Return 0 on success and -1 on error.
 *
 * If the first token on the stream is a '[' then we remove this token
 * from the stream and keep track of the fact that the sequence
 * is given in flow format.
 * Otherwise, we assume the first token is the dash that introduces
 * the first element of the sequence and keep track of its indentation,
 * but keep the token on the stream.
 * In both cases, the next token we expect is the first element
 * of the sequence.
 */
int isl_stream_yaml_read_start_sequence(__isl_keep isl_stream *s)
{
	struct isl_token *tok;
	int indent;

	if (push_state(s, isl_yaml_sequence_start) < 0)
		return -1;

	tok = isl_stream_next_token(s);
	if (!tok) {
		if (s->eof)
			isl_stream_error(s, NULL, "unexpected EOF");
		return -1;
	}
	if (isl_token_get_type(tok) == '[') {
		isl_token_free(tok);
		return set_yaml_indent(s, ISL_YAML_INDENT_FLOW);
	}
	indent = tok->col - 1;
	isl_stream_push_token(s, tok);

	return set_yaml_indent(s, indent);
}

/* Finish reading a YAML sequence.
 * Return 0 on success and -1 on error.
 *
 * If the sequence started with a '[', then we expect a ']' to close
 * the sequence.
 * Otherwise, we double-check that the next token (if any)
 * is not a dash or that it has a smaller indentation than
 * that of the current sequence.
 */
int isl_stream_yaml_read_end_sequence(__isl_keep isl_stream *s)
{
	struct isl_token *tok;
	int indent;
	int dash;

	if (get_yaml_indent(s) == ISL_YAML_INDENT_FLOW) {
		if (isl_stream_eat(s, ']') < 0)
			return -1;
		return pop_state(s);
	}

	tok = isl_stream_next_token(s);
	if (!tok)
		return pop_state(s);

	indent = tok->col - 1;
	dash = tok->type == '-';
	isl_stream_push_token(s, tok);

	if (indent >= get_yaml_indent(s) && dash)
		isl_die(isl_stream_get_ctx(s), isl_error_invalid,
			"sequence not finished", return -1);

	return pop_state(s);
}
