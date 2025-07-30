#include <string.h>
#include <isl_int.h>
#include <isl_printer_private.h>

static __isl_give isl_printer *file_start_line(__isl_take isl_printer *p)
{
	fprintf(p->file, "%s%*s%s", p->indent_prefix ? p->indent_prefix : "",
				    p->indent, "", p->prefix ? p->prefix : "");
	return p;
}

static __isl_give isl_printer *file_end_line(__isl_take isl_printer *p)
{
	fprintf(p->file, "%s\n", p->suffix ? p->suffix : "");
	return p;
}

static __isl_give isl_printer *file_flush(__isl_take isl_printer *p)
{
	fflush(p->file);
	return p;
}

static __isl_give isl_printer *file_print_str(__isl_take isl_printer *p,
	const char *s)
{
	fprintf(p->file, "%s", s);
	return p;
}

static __isl_give isl_printer *file_print_double(__isl_take isl_printer *p,
	double d)
{
	fprintf(p->file, "%g", d);
	return p;
}

static __isl_give isl_printer *file_print_int(__isl_take isl_printer *p, int i)
{
	fprintf(p->file, "%d", i);
	return p;
}

static __isl_give isl_printer *file_print_isl_int(__isl_take isl_printer *p, isl_int i)
{
	isl_int_print(p->file, i, p->width);
	return p;
}

static int grow_buf(__isl_keep isl_printer *p, int extra)
{
	int new_size;
	char *new_buf;

	if (p->buf_size == 0)
		return -1;

	new_size = ((p->buf_n + extra + 1) * 3) / 2;
	new_buf = isl_realloc_array(p->ctx, p->buf, char, new_size);
	if (!new_buf) {
		p->buf_size = 0;
		return -1;
	}
	p->buf = new_buf;
	p->buf_size = new_size;

	return 0;
}

static __isl_give isl_printer *str_print(__isl_take isl_printer *p,
	const char *s, int len)
{
	if (p->buf_n + len + 1 >= p->buf_size && grow_buf(p, len))
		goto error;
	memcpy(p->buf + p->buf_n, s, len);
	p->buf_n += len;

	p->buf[p->buf_n] = '\0';
	return p;
error:
	isl_printer_free(p);
	return NULL;
}

static __isl_give isl_printer *str_print_indent(__isl_take isl_printer *p,
	int indent)
{
	int i;

	if (p->buf_n + indent + 1 >= p->buf_size && grow_buf(p, indent))
		goto error;
	for (i = 0; i < indent; ++i)
		p->buf[p->buf_n++] = ' ';
	return p;
error:
	isl_printer_free(p);
	return NULL;
}

static __isl_give isl_printer *str_start_line(__isl_take isl_printer *p)
{
	if (p->indent_prefix)
		p = str_print(p, p->indent_prefix, strlen(p->indent_prefix));
	p = str_print_indent(p, p->indent);
	if (p->prefix)
		p = str_print(p, p->prefix, strlen(p->prefix));
	return p;
}

static __isl_give isl_printer *str_end_line(__isl_take isl_printer *p)
{
	if (p->suffix)
		p = str_print(p, p->suffix, strlen(p->suffix));
	p = str_print(p, "\n", strlen("\n"));
	return p;
}

static __isl_give isl_printer *str_flush(__isl_take isl_printer *p)
{
	p->buf_n = 0;
	p->buf[p->buf_n] = '\0';
	return p;
}

static __isl_give isl_printer *str_print_str(__isl_take isl_printer *p,
	const char *s)
{
	return str_print(p, s, strlen(s));
}

static __isl_give isl_printer *str_print_double(__isl_take isl_printer *p,
	double d)
{
	int left = p->buf_size - p->buf_n;
	int need = snprintf(p->buf + p->buf_n, left, "%g", d);
	if (need >= left) {
		if (grow_buf(p, need))
			goto error;
		left = p->buf_size - p->buf_n;
		need = snprintf(p->buf + p->buf_n, left, "%g", d);
	}
	p->buf_n += need;
	return p;
error:
	isl_printer_free(p);
	return NULL;
}

static __isl_give isl_printer *str_print_int(__isl_take isl_printer *p, int i)
{
	int left = p->buf_size - p->buf_n;
	int need = snprintf(p->buf + p->buf_n, left, "%d", i);
	if (need >= left) {
		if (grow_buf(p, need))
			goto error;
		left = p->buf_size - p->buf_n;
		need = snprintf(p->buf + p->buf_n, left, "%d", i);
	}
	p->buf_n += need;
	return p;
error:
	isl_printer_free(p);
	return NULL;
}

static __isl_give isl_printer *str_print_isl_int(__isl_take isl_printer *p,
	isl_int i)
{
	char *s;
	int len;

	s = isl_int_get_str(i);
	len = strlen(s);
	if (len < p->width)
		p = str_print_indent(p, p->width - len);
	p = str_print(p, s, len);
	isl_int_free_str(s);
	return p;
}

struct isl_printer_ops {
	__isl_give isl_printer *(*start_line)(__isl_take isl_printer *p);
	__isl_give isl_printer *(*end_line)(__isl_take isl_printer *p);
	__isl_give isl_printer *(*print_double)(__isl_take isl_printer *p,
		double d);
	__isl_give isl_printer *(*print_int)(__isl_take isl_printer *p, int i);
	__isl_give isl_printer *(*print_isl_int)(__isl_take isl_printer *p,
						isl_int i);
	__isl_give isl_printer *(*print_str)(__isl_take isl_printer *p,
						const char *s);
	__isl_give isl_printer *(*flush)(__isl_take isl_printer *p);
};

static struct isl_printer_ops file_ops = {
	file_start_line,
	file_end_line,
	file_print_double,
	file_print_int,
	file_print_isl_int,
	file_print_str,
	file_flush
};

static struct isl_printer_ops str_ops = {
	str_start_line,
	str_end_line,
	str_print_double,
	str_print_int,
	str_print_isl_int,
	str_print_str,
	str_flush
};

__isl_give isl_printer *isl_printer_to_file(isl_ctx *ctx, FILE *file)
{
	struct isl_printer *p = isl_calloc_type(ctx, struct isl_printer);
	if (!p)
		return NULL;
	p->ctx = ctx;
	isl_ctx_ref(p->ctx);
	p->ops = &file_ops;
	p->file = file;
	p->buf = NULL;
	p->buf_n = 0;
	p->buf_size = 0;
	p->indent = 0;
	p->output_format = ISL_FORMAT_ISL;
	p->indent_prefix = NULL;
	p->prefix = NULL;
	p->suffix = NULL;
	p->width = 0;
	p->yaml_style = ISL_YAML_STYLE_FLOW;

	return p;
}

__isl_give isl_printer *isl_printer_to_str(isl_ctx *ctx)
{
	struct isl_printer *p = isl_calloc_type(ctx, struct isl_printer);
	if (!p)
		return NULL;
	p->ctx = ctx;
	isl_ctx_ref(p->ctx);
	p->ops = &str_ops;
	p->file = NULL;
	p->buf = isl_alloc_array(ctx, char, 256);
	if (!p->buf)
		goto error;
	p->buf_n = 0;
	p->buf[0] = '\0';
	p->buf_size = 256;
	p->indent = 0;
	p->output_format = ISL_FORMAT_ISL;
	p->indent_prefix = NULL;
	p->prefix = NULL;
	p->suffix = NULL;
	p->width = 0;
	p->yaml_style = ISL_YAML_STYLE_FLOW;

	return p;
error:
	isl_printer_free(p);
	return NULL;
}

__isl_null isl_printer *isl_printer_free(__isl_take isl_printer *p)
{
	if (!p)
		return NULL;
	free(p->buf);
	free(p->indent_prefix);
	free(p->prefix);
	free(p->suffix);
	free(p->yaml_state);
	isl_id_to_id_free(p->notes);
	isl_ctx_deref(p->ctx);
	free(p);

	return NULL;
}

isl_ctx *isl_printer_get_ctx(__isl_keep isl_printer *printer)
{
	return printer ? printer->ctx : NULL;
}

FILE *isl_printer_get_file(__isl_keep isl_printer *printer)
{
	if (!printer)
		return NULL;
	if (!printer->file)
		isl_die(isl_printer_get_ctx(printer), isl_error_invalid,
			"not a file printer", return NULL);
	return printer->file;
}

__isl_give isl_printer *isl_printer_set_isl_int_width(__isl_take isl_printer *p,
	int width)
{
	if (!p)
		return NULL;

	p->width = width;

	return p;
}

__isl_give isl_printer *isl_printer_set_indent(__isl_take isl_printer *p,
	int indent)
{
	if (!p)
		return NULL;

	p->indent = indent;

	return p;
}

__isl_give isl_printer *isl_printer_indent(__isl_take isl_printer *p,
	int indent)
{
	if (!p)
		return NULL;

	p->indent += indent;
	if (p->indent < 0)
		p->indent = 0;

	return p;
}

/* Replace the indent prefix of "p" by "prefix".
 */
__isl_give isl_printer *isl_printer_set_indent_prefix(__isl_take isl_printer *p,
	const char *prefix)
{
	if (!p)
		return NULL;

	free(p->indent_prefix);
	p->indent_prefix = prefix ? strdup(prefix) : NULL;

	return p;
}

__isl_give isl_printer *isl_printer_set_prefix(__isl_take isl_printer *p,
	const char *prefix)
{
	if (!p)
		return NULL;

	free(p->prefix);
	p->prefix = prefix ? strdup(prefix) : NULL;

	return p;
}

__isl_give isl_printer *isl_printer_set_suffix(__isl_take isl_printer *p,
	const char *suffix)
{
	if (!p)
		return NULL;

	free(p->suffix);
	p->suffix = suffix ? strdup(suffix) : NULL;

	return p;
}

__isl_give isl_printer *isl_printer_set_output_format(__isl_take isl_printer *p,
	int output_format)
{
	if (!p)
		return NULL;

	p->output_format = output_format;

	return p;
}

int isl_printer_get_output_format(__isl_keep isl_printer *p)
{
	if (!p)
		return -1;
	return p->output_format;
}

/* Does "p" have a note with identifier "id"?
 */
isl_bool isl_printer_has_note(__isl_keep isl_printer *p,
	__isl_keep isl_id *id)
{
	if (!p || !id)
		return isl_bool_error;
	if (!p->notes)
		return isl_bool_false;
	return isl_id_to_id_has(p->notes, id);
}

/* Retrieve the note identified by "id" from "p".
 * The note is assumed to exist.
 */
__isl_give isl_id *isl_printer_get_note(__isl_keep isl_printer *p,
	__isl_take isl_id *id)
{
	isl_bool has_note;

	has_note = isl_printer_has_note(p, id);
	if (has_note < 0)
		goto error;
	if (!has_note)
		isl_die(isl_printer_get_ctx(p), isl_error_invalid,
			"no such note", goto error);

	return isl_id_to_id_get(p->notes, id);
error:
	isl_id_free(id);
	return NULL;
}

/* Associate "note" to the identifier "id" in "p",
 * replacing the previous note associated to the identifier, if any.
 */
__isl_give isl_printer *isl_printer_set_note(__isl_take isl_printer *p,
	__isl_take isl_id *id, __isl_take isl_id *note)
{
	if (!p || !id || !note)
		goto error;
	if (!p->notes) {
		p->notes = isl_id_to_id_alloc(isl_printer_get_ctx(p), 1);
		if (!p->notes)
			goto error;
	}
	p->notes = isl_id_to_id_set(p->notes, id, note);
	if (!p->notes)
		return isl_printer_free(p);
	return p;
error:
	isl_printer_free(p);
	isl_id_free(id);
	isl_id_free(note);
	return NULL;
}

/* Keep track of whether the printing to "p" is being performed from
 * an isl_*_dump function as specified by "dump".
 */
__isl_give isl_printer *isl_printer_set_dump(__isl_take isl_printer *p,
	int dump)
{
	if (!p)
		return NULL;

	p->dump = dump;

	return p;
}

/* Set the YAML style of "p" to "yaml_style" and return the updated printer.
 */
__isl_give isl_printer *isl_printer_set_yaml_style(__isl_take isl_printer *p,
	int yaml_style)
{
	if (!p)
		return NULL;

	p->yaml_style = yaml_style;

	return p;
}

/* Return the YAML style of "p" or -1 on error.
 */
int isl_printer_get_yaml_style(__isl_keep isl_printer *p)
{
	if (!p)
		return -1;
	return p->yaml_style;
}

/* Push "state" onto the stack of currently active YAML elements and
 * return the updated printer.
 */
static __isl_give isl_printer *push_state(__isl_take isl_printer *p,
	enum isl_yaml_state state)
{
	if (!p)
		return NULL;

	if (p->yaml_size < p->yaml_depth + 1) {
		enum isl_yaml_state *state;
		state = isl_realloc_array(p->ctx, p->yaml_state,
					enum isl_yaml_state, p->yaml_depth + 1);
		if (!state)
			return isl_printer_free(p);
		p->yaml_state = state;
		p->yaml_size = p->yaml_depth + 1;
	}

	p->yaml_state[p->yaml_depth] = state;
	p->yaml_depth++;

	return p;
}

/* Remove the innermost active YAML element from the stack and
 * return the updated printer.
 */
static __isl_give isl_printer *pop_state(__isl_take isl_printer *p)
{
	if (!p)
		return NULL;
	p->yaml_depth--;
	return p;
}

/* Set the state of the innermost active YAML element to "state" and
 * return the updated printer.
 */
static __isl_give isl_printer *update_state(__isl_take isl_printer *p,
	enum isl_yaml_state state)
{
	if (!p)
		return NULL;
	if (p->yaml_depth < 1)
		isl_die(isl_printer_get_ctx(p), isl_error_invalid,
			"not in YAML construct", return isl_printer_free(p));

	p->yaml_state[p->yaml_depth - 1] = state;

	return p;
}

/* Return the state of the innermost active YAML element.
 * Return isl_yaml_none if we are not inside any YAML element.
 */
static enum isl_yaml_state current_state(__isl_keep isl_printer *p)
{
	if (!p)
		return isl_yaml_none;
	if (p->yaml_depth < 1)
		return isl_yaml_none;
	return p->yaml_state[p->yaml_depth - 1];
}

/* If we are printing a YAML document and we are at the start of an element,
 * print whatever is needed before we can print the actual element and
 * keep track of the fact that we are now printing the element.
 * If "eol" is set, then whatever we print is going to be the last
 * thing that gets printed on this line.
 *
 * If we are about the print the first key of a mapping, then nothing
 * extra needs to be printed.  For any other key, however, we need
 * to either move to the next line (in block format) or print a comma
 * (in flow format).
 * Before printing a value in a mapping, we need to print a colon.
 *
 * For sequences, in flow format, we only need to print a comma
 * for each element except the first.
 * In block format, before the first element in the sequence,
 * we move to a new line, print a dash and increase the indentation.
 * Before any other element, we print a dash on a new line,
 * temporarily moving the indentation back.
 */
static __isl_give isl_printer *enter_state(__isl_take isl_printer *p,
	int eol)
{
	enum isl_yaml_state state;

	if (!p)
		return NULL;

	state = current_state(p);
	if (state == isl_yaml_mapping_val_start) {
		if (eol)
			p = p->ops->print_str(p, ":");
		else
			p = p->ops->print_str(p, ": ");
		p = update_state(p, isl_yaml_mapping_val);
	} else if (state == isl_yaml_mapping_first_key_start) {
		p = update_state(p, isl_yaml_mapping_key);
	} else if (state == isl_yaml_mapping_key_start) {
		if (p->yaml_style == ISL_YAML_STYLE_FLOW)
			p = p->ops->print_str(p, ", ");
		else {
			p = p->ops->end_line(p);
			p = p->ops->start_line(p);
		}
		p = update_state(p, isl_yaml_mapping_key);
	} else if (state == isl_yaml_sequence_first_start) {
		if (p->yaml_style != ISL_YAML_STYLE_FLOW) {
			p = p->ops->end_line(p);
			p = p->ops->start_line(p);
			p = p->ops->print_str(p, "- ");
			p = isl_printer_indent(p, 2);
		}
		p = update_state(p, isl_yaml_sequence);
	} else if (state == isl_yaml_sequence_start) {
		if (p->yaml_style == ISL_YAML_STYLE_FLOW)
			p = p->ops->print_str(p, ", ");
		else {
			p = p->ops->end_line(p);
			p = isl_printer_indent(p, -2);
			p = p->ops->start_line(p);
			p = p->ops->print_str(p, "- ");
			p = isl_printer_indent(p, 2);
		}
		p = update_state(p, isl_yaml_sequence);
	}

	return p;
}

__isl_give isl_printer *isl_printer_print_str(__isl_take isl_printer *p,
	const char *s)
{
	if (!p)
		return NULL;
	if (!s)
		return isl_printer_free(p);
	p = enter_state(p, 0);
	if (!p)
		return NULL;
	return p->ops->print_str(p, s);
}

__isl_give isl_printer *isl_printer_print_double(__isl_take isl_printer *p,
	double d)
{
	p = enter_state(p, 0);
	if (!p)
		return NULL;

	return p->ops->print_double(p, d);
}

__isl_give isl_printer *isl_printer_print_int(__isl_take isl_printer *p, int i)
{
	p = enter_state(p, 0);
	if (!p)
		return NULL;

	return p->ops->print_int(p, i);
}

__isl_give isl_printer *isl_printer_print_isl_int(__isl_take isl_printer *p,
	isl_int i)
{
	p = enter_state(p, 0);
	if (!p)
		return NULL;

	return p->ops->print_isl_int(p, i);
}

__isl_give isl_printer *isl_printer_start_line(__isl_take isl_printer *p)
{
	if (!p)
		return NULL;

	return p->ops->start_line(p);
}

__isl_give isl_printer *isl_printer_end_line(__isl_take isl_printer *p)
{
	if (!p)
		return NULL;

	return p->ops->end_line(p);
}

/* Return a copy of the string constructed by the string printer "printer".
 */
__isl_give char *isl_printer_get_str(__isl_keep isl_printer *printer)
{
	if (!printer)
		return NULL;
	if (printer->ops != &str_ops)
		isl_die(isl_printer_get_ctx(printer), isl_error_invalid,
			"isl_printer_get_str can only be called on a string "
			"printer", return NULL);
	if (!printer->buf)
		return NULL;
	return strdup(printer->buf);
}

__isl_give isl_printer *isl_printer_flush(__isl_take isl_printer *p)
{
	if (!p)
		return NULL;

	return p->ops->flush(p);
}

/* Start a YAML mapping and push a new state to reflect that we
 * are about to print the first key in a mapping.
 *
 * In flow style, print the opening brace.
 * In block style, move to the next line with an increased indentation,
 * except if this is the outer mapping or if we are inside a sequence
 * (in which case we have already increased the indentation and we want
 * to print the first key on the same line as the dash).
 */
__isl_give isl_printer *isl_printer_yaml_start_mapping(
	__isl_take isl_printer *p)
{
	enum isl_yaml_state state;

	if (!p)
		return NULL;
	p = enter_state(p, p->yaml_style == ISL_YAML_STYLE_BLOCK);
	if (!p)
		return NULL;
	state = current_state(p);
	if (p->yaml_style == ISL_YAML_STYLE_FLOW)
		p = p->ops->print_str(p, "{ ");
	else if (state != isl_yaml_none && state != isl_yaml_sequence) {
		p = p->ops->end_line(p);
		p = isl_printer_indent(p, 2);
		p = p->ops->start_line(p);
	}
	p = push_state(p, isl_yaml_mapping_first_key_start);
	return p;
}

/* Finish a YAML mapping and pop it from the state stack.
 *
 * In flow style, print the closing brace.
 *
 * In block style, first check if we are still in the
 * isl_yaml_mapping_first_key_start state.  If so, we have not printed
 * anything yet, so print "{}" to indicate an empty mapping.
 * If we increased the indentation in isl_printer_yaml_start_mapping,
 * then decrease it again.
 * If this is the outer mapping then print a newline.
 */
__isl_give isl_printer *isl_printer_yaml_end_mapping(
	__isl_take isl_printer *p)
{
	enum isl_yaml_state state;

	state = current_state(p);
	p = pop_state(p);
	if (!p)
		return NULL;
	if (p->yaml_style == ISL_YAML_STYLE_FLOW)
		return p->ops->print_str(p, " }");
	if (state == isl_yaml_mapping_first_key_start)
		p = p->ops->print_str(p, "{}");
	if (!p)
		return NULL;
	state = current_state(p);
	if (state != isl_yaml_none && state != isl_yaml_sequence)
		p = isl_printer_indent(p, -2);
	if (state == isl_yaml_none)
		p = p->ops->end_line(p);
	return p;
}

/* Start a YAML sequence and push a new state to reflect that we
 * are about to print the first element in a sequence.
 *
 * In flow style, print the opening bracket.
 */
__isl_give isl_printer *isl_printer_yaml_start_sequence(
	__isl_take isl_printer *p)
{
	if (!p)
		return NULL;
	p = enter_state(p, p->yaml_style == ISL_YAML_STYLE_BLOCK);
	p = push_state(p, isl_yaml_sequence_first_start);
	if (!p)
		return NULL;
	if (p->yaml_style == ISL_YAML_STYLE_FLOW)
		p = p->ops->print_str(p, "[ ");
	return p;
}

/* Finish a YAML sequence and pop it from the state stack.
 *
 * In flow style, print the closing bracket.
 *
 * In block style, check if we are still in the
 * isl_yaml_sequence_first_start state.  If so, we have not printed
 * anything yet, so print "[]" or " []" to indicate an empty sequence.
 * We print the extra space when we instructed enter_state not
 * to print a space at the end of the line.
 * Otherwise, undo the increase in indentation performed by
 * enter_state when moving away from the isl_yaml_sequence_first_start
 * state.
 * If this is the outer sequence then print a newline.
 */
__isl_give isl_printer *isl_printer_yaml_end_sequence(
	__isl_take isl_printer *p)
{
	enum isl_yaml_state state, up;

	state = current_state(p);
	p = pop_state(p);
	if (!p)
		return NULL;
	if (p->yaml_style == ISL_YAML_STYLE_FLOW)
		return p->ops->print_str(p, " ]");
	up = current_state(p);
	if (state == isl_yaml_sequence_first_start) {
		if (up == isl_yaml_mapping_val)
			p = p->ops->print_str(p, " []");
		else
			p = p->ops->print_str(p, "[]");
	} else {
		p = isl_printer_indent(p, -2);
	}
	if (!p)
		return NULL;
	state = current_state(p);
	if (state == isl_yaml_none)
		p = p->ops->end_line(p);
	return p;
}

/* Mark the fact that the current element is finished and that
 * the next output belongs to the next element.
 * In particular, if we are printing a key, then prepare for
 * printing the subsequent value.  If we are printing a value,
 * prepare for printing the next key.  If we are printing an
 * element in a sequence, prepare for printing the next element.
 */
__isl_give isl_printer *isl_printer_yaml_next(__isl_take isl_printer *p)
{
	enum isl_yaml_state state;

	if (!p)
		return NULL;
	if (p->yaml_depth < 1)
		isl_die(isl_printer_get_ctx(p), isl_error_invalid,
			"not in YAML construct", return isl_printer_free(p));

	state = current_state(p);
	if (state == isl_yaml_mapping_key)
		state = isl_yaml_mapping_val_start;
	else if (state == isl_yaml_mapping_val)
		state = isl_yaml_mapping_key_start;
	else if (state == isl_yaml_sequence)
		state = isl_yaml_sequence_start;
	p = update_state(p, state);

	return p;
}
