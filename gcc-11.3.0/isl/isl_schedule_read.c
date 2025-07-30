#include <string.h>

#include <isl/schedule.h>
#include <isl/stream.h>
#include <isl_schedule_private.h>
#include <isl_schedule_tree.h>

/* An enumeration of the various keys that may appear in a YAML mapping
 * of a schedule.
 */
enum isl_schedule_key {
	isl_schedule_key_error = -1,
	isl_schedule_key_child,
	isl_schedule_key_coincident,
	isl_schedule_key_context,
	isl_schedule_key_contraction,
	isl_schedule_key_domain,
	isl_schedule_key_expansion,
	isl_schedule_key_extension,
	isl_schedule_key_filter,
	isl_schedule_key_guard,
	isl_schedule_key_leaf,
	isl_schedule_key_mark,
	isl_schedule_key_options,
	isl_schedule_key_permutable,
	isl_schedule_key_schedule,
	isl_schedule_key_sequence,
	isl_schedule_key_set
};

/* Extract a mapping key from the token "tok".
 * Return isl_schedule_key_error on error, i.e., if "tok" does not
 * correspond to any known key.
 */
static enum isl_schedule_key extract_key(__isl_keep isl_stream *s,
	struct isl_token *tok)
{
	int type;
	char *name;
	enum isl_schedule_key key;
	isl_ctx *ctx;

	ctx = isl_stream_get_ctx(s);
	type = isl_token_get_type(tok);
	if (type != ISL_TOKEN_IDENT && type != ISL_TOKEN_STRING) {
		isl_stream_error(s, tok, "expecting key");
		return isl_schedule_key_error;
	}
	name = isl_token_get_str(ctx, tok);
	if (!strcmp(name, "child"))
		key = isl_schedule_key_child;
	else if (!strcmp(name, "coincident"))
		key = isl_schedule_key_coincident;
	else if (!strcmp(name, "context"))
		key = isl_schedule_key_context;
	else if (!strcmp(name, "contraction"))
		key = isl_schedule_key_contraction;
	else if (!strcmp(name, "domain"))
		key = isl_schedule_key_domain;
	else if (!strcmp(name, "expansion"))
		key = isl_schedule_key_expansion;
	else if (!strcmp(name, "extension"))
		key = isl_schedule_key_extension;
	else if (!strcmp(name, "filter"))
		key = isl_schedule_key_filter;
	else if (!strcmp(name, "guard"))
		key = isl_schedule_key_guard;
	else if (!strcmp(name, "leaf"))
		key = isl_schedule_key_leaf;
	else if (!strcmp(name, "mark"))
		key = isl_schedule_key_mark;
	else if (!strcmp(name, "options"))
		key = isl_schedule_key_options;
	else if (!strcmp(name, "schedule"))
		key = isl_schedule_key_schedule;
	else if (!strcmp(name, "sequence"))
		key = isl_schedule_key_sequence;
	else if (!strcmp(name, "set"))
		key = isl_schedule_key_set;
	else if (!strcmp(name, "permutable"))
		key = isl_schedule_key_permutable;
	else
		isl_die(ctx, isl_error_invalid, "unknown key",
			key = isl_schedule_key_error);
	free(name);
	return key;
}

/* Read a key from "s" and return the corresponding enum.
 * Return isl_schedule_key_error on error, i.e., if the first token
 * on the stream does not correspond to any known key.
 */
static enum isl_schedule_key get_key(__isl_keep isl_stream *s)
{
	struct isl_token *tok;
	enum isl_schedule_key key;

	tok = isl_stream_next_token(s);
	key = extract_key(s, tok);
	isl_token_free(tok);

	return key;
}

static __isl_give isl_schedule_tree *isl_stream_read_schedule_tree(
	__isl_keep isl_stream *s);

/* Read a subtree with context root node from "s".
 */
static __isl_give isl_schedule_tree *read_context(__isl_keep isl_stream *s)
{
	isl_set *context = NULL;
	isl_schedule_tree *tree;
	isl_ctx *ctx;
	struct isl_token *tok;
	enum isl_schedule_key key;
	char *str;
	int more;

	ctx = isl_stream_get_ctx(s);

	key = get_key(s);

	if (isl_stream_yaml_next(s) < 0)
		return NULL;

	tok = isl_stream_next_token(s);
	if (!tok) {
		isl_stream_error(s, NULL, "unexpected EOF");
		return NULL;
	}
	str = isl_token_get_str(ctx, tok);
	context = isl_set_read_from_str(ctx, str);
	free(str);
	isl_token_free(tok);

	more = isl_stream_yaml_next(s);
	if (more < 0)
		goto error;
	if (!more) {
		tree = isl_schedule_tree_from_context(context);
	} else {
		key = get_key(s);
		if (key != isl_schedule_key_child)
			isl_die(ctx, isl_error_invalid, "expecting child",
				goto error);
		if (isl_stream_yaml_next(s) < 0)
			goto error;
		tree = isl_stream_read_schedule_tree(s);
		tree = isl_schedule_tree_insert_context(tree, context);
	}

	return tree;
error:
	isl_set_free(context);
	return NULL;
}

/* Read a subtree with domain root node from "s".
 */
static __isl_give isl_schedule_tree *read_domain(__isl_keep isl_stream *s)
{
	isl_union_set *domain = NULL;
	isl_schedule_tree *tree;
	isl_ctx *ctx;
	struct isl_token *tok;
	enum isl_schedule_key key;
	char *str;
	int more;

	ctx = isl_stream_get_ctx(s);

	key = get_key(s);

	if (isl_stream_yaml_next(s) < 0)
		return NULL;

	tok = isl_stream_next_token(s);
	if (!tok) {
		isl_stream_error(s, NULL, "unexpected EOF");
		return NULL;
	}
	str = isl_token_get_str(ctx, tok);
	domain = isl_union_set_read_from_str(ctx, str);
	free(str);
	isl_token_free(tok);

	more = isl_stream_yaml_next(s);
	if (more < 0)
		goto error;
	if (!more) {
		tree = isl_schedule_tree_from_domain(domain);
	} else {
		key = get_key(s);
		if (key != isl_schedule_key_child)
			isl_die(ctx, isl_error_invalid, "expecting child",
				goto error);
		if (isl_stream_yaml_next(s) < 0)
			goto error;
		tree = isl_stream_read_schedule_tree(s);
		tree = isl_schedule_tree_insert_domain(tree, domain);
	}

	return tree;
error:
	isl_union_set_free(domain);
	return NULL;
}

/* Read a subtree with expansion root node from "s".
 */
static __isl_give isl_schedule_tree *read_expansion(isl_stream *s)
{
	isl_ctx *ctx;
	isl_union_pw_multi_aff *contraction = NULL;
	isl_union_map *expansion = NULL;
	isl_schedule_tree *tree = NULL;
	int more;

	ctx = isl_stream_get_ctx(s);

	do {
		struct isl_token *tok;
		enum isl_schedule_key key;
		char *str;

		key = get_key(s);
		if (isl_stream_yaml_next(s) < 0)
			goto error;

		switch (key) {
		case isl_schedule_key_contraction:
			isl_union_pw_multi_aff_free(contraction);
			tok = isl_stream_next_token(s);
			str = isl_token_get_str(ctx, tok);
			contraction = isl_union_pw_multi_aff_read_from_str(ctx,
									str);
			free(str);
			isl_token_free(tok);
			if (!contraction)
				goto error;
			break;
		case isl_schedule_key_expansion:
			isl_union_map_free(expansion);
			tok = isl_stream_next_token(s);
			str = isl_token_get_str(ctx, tok);
			expansion = isl_union_map_read_from_str(ctx, str);
			free(str);
			isl_token_free(tok);
			if (!expansion)
				goto error;
			break;
		case isl_schedule_key_child:
			isl_schedule_tree_free(tree);
			tree = isl_stream_read_schedule_tree(s);
			if (!tree)
				goto error;
			break;
		default:
			isl_die(ctx, isl_error_invalid, "unexpected key",
				goto error);
		}
	} while ((more = isl_stream_yaml_next(s)) > 0);

	if (more < 0)
		goto error;

	if (!contraction)
		isl_die(ctx, isl_error_invalid, "missing contraction",
			goto error);
	if (!expansion)
		isl_die(ctx, isl_error_invalid, "missing expansion",
			goto error);

	if (!tree)
		return isl_schedule_tree_from_expansion(contraction, expansion);
	return isl_schedule_tree_insert_expansion(tree, contraction, expansion);
error:
	isl_schedule_tree_free(tree);
	isl_union_pw_multi_aff_free(contraction);
	isl_union_map_free(expansion);
	return NULL;
}

/* Read a subtree with extension root node from "s".
 */
static __isl_give isl_schedule_tree *read_extension(isl_stream *s)
{
	isl_union_map *extension = NULL;
	isl_schedule_tree *tree;
	isl_ctx *ctx;
	struct isl_token *tok;
	enum isl_schedule_key key;
	char *str;
	int more;

	ctx = isl_stream_get_ctx(s);

	key = get_key(s);

	if (isl_stream_yaml_next(s) < 0)
		return NULL;

	tok = isl_stream_next_token(s);
	if (!tok) {
		isl_stream_error(s, NULL, "unexpected EOF");
		return NULL;
	}
	str = isl_token_get_str(ctx, tok);
	extension = isl_union_map_read_from_str(ctx, str);
	free(str);
	isl_token_free(tok);

	more = isl_stream_yaml_next(s);
	if (more < 0)
		goto error;
	if (!more) {
		tree = isl_schedule_tree_from_extension(extension);
	} else {
		key = get_key(s);
		if (key != isl_schedule_key_child)
			isl_die(ctx, isl_error_invalid, "expecting child",
				goto error);
		if (isl_stream_yaml_next(s) < 0)
			goto error;
		tree = isl_stream_read_schedule_tree(s);
		tree = isl_schedule_tree_insert_extension(tree, extension);
	}

	return tree;
error:
	isl_union_map_free(extension);
	return NULL;
}

/* Read a subtree with filter root node from "s".
 */
static __isl_give isl_schedule_tree *read_filter(__isl_keep isl_stream *s)
{
	isl_union_set *filter = NULL;
	isl_schedule_tree *tree;
	isl_ctx *ctx;
	struct isl_token *tok;
	enum isl_schedule_key key;
	char *str;
	int more;

	ctx = isl_stream_get_ctx(s);

	key = get_key(s);

	if (isl_stream_yaml_next(s) < 0)
		return NULL;

	tok = isl_stream_next_token(s);
	if (!tok) {
		isl_stream_error(s, NULL, "unexpected EOF");
		return NULL;
	}
	str = isl_token_get_str(ctx, tok);
	filter = isl_union_set_read_from_str(ctx, str);
	free(str);
	isl_token_free(tok);

	more = isl_stream_yaml_next(s);
	if (more < 0)
		goto error;
	if (!more) {
		tree = isl_schedule_tree_from_filter(filter);
	} else {
		key = get_key(s);
		if (key != isl_schedule_key_child)
			isl_die(ctx, isl_error_invalid, "expecting child",
				goto error);
		if (isl_stream_yaml_next(s) < 0)
			goto error;
		tree = isl_stream_read_schedule_tree(s);
		tree = isl_schedule_tree_insert_filter(tree, filter);
	}

	return tree;
error:
	isl_union_set_free(filter);
	return NULL;
}

/* Read a subtree with guard root node from "s".
 */
static __isl_give isl_schedule_tree *read_guard(isl_stream *s)
{
	isl_set *guard = NULL;
	isl_schedule_tree *tree;
	isl_ctx *ctx;
	struct isl_token *tok;
	enum isl_schedule_key key;
	char *str;
	int more;

	ctx = isl_stream_get_ctx(s);

	key = get_key(s);

	if (isl_stream_yaml_next(s) < 0)
		return NULL;

	tok = isl_stream_next_token(s);
	if (!tok) {
		isl_stream_error(s, NULL, "unexpected EOF");
		return NULL;
	}
	str = isl_token_get_str(ctx, tok);
	guard = isl_set_read_from_str(ctx, str);
	free(str);
	isl_token_free(tok);

	more = isl_stream_yaml_next(s);
	if (more < 0)
		goto error;
	if (!more) {
		tree = isl_schedule_tree_from_guard(guard);
	} else {
		key = get_key(s);
		if (key != isl_schedule_key_child)
			isl_die(ctx, isl_error_invalid, "expecting child",
				goto error);
		if (isl_stream_yaml_next(s) < 0)
			goto error;
		tree = isl_stream_read_schedule_tree(s);
		tree = isl_schedule_tree_insert_guard(tree, guard);
	}

	return tree;
error:
	isl_set_free(guard);
	return NULL;
}

/* Read a subtree with mark root node from "s".
 */
static __isl_give isl_schedule_tree *read_mark(isl_stream *s)
{
	isl_id *mark;
	isl_schedule_tree *tree;
	isl_ctx *ctx;
	struct isl_token *tok;
	enum isl_schedule_key key;
	char *str;
	int more;

	ctx = isl_stream_get_ctx(s);

	key = get_key(s);

	if (isl_stream_yaml_next(s) < 0)
		return NULL;

	tok = isl_stream_next_token(s);
	if (!tok) {
		isl_stream_error(s, NULL, "unexpected EOF");
		return NULL;
	}
	str = isl_token_get_str(ctx, tok);
	mark = isl_id_alloc(ctx, str, NULL);
	free(str);
	isl_token_free(tok);

	more = isl_stream_yaml_next(s);
	if (more < 0)
		goto error;
	if (!more) {
		isl_die(ctx, isl_error_invalid, "expecting child",
			goto error);
	} else {
		key = get_key(s);
		if (key != isl_schedule_key_child)
			isl_die(ctx, isl_error_invalid, "expecting child",
				goto error);
		if (isl_stream_yaml_next(s) < 0)
			goto error;
		tree = isl_stream_read_schedule_tree(s);
		tree = isl_schedule_tree_insert_mark(tree, mark);
	}

	return tree;
error:
	isl_id_free(mark);
	return NULL;
}

/* Read a sequence of integers from "s" (representing the coincident
 * property of a band node).
 */
static __isl_give isl_val_list *read_coincident(__isl_keep isl_stream *s)
{
	isl_ctx *ctx;
	isl_val_list *list;
	int more;

	ctx = isl_stream_get_ctx(s);

	if (isl_stream_yaml_read_start_sequence(s) < 0)
		return NULL;

	list = isl_val_list_alloc(ctx, 0);
	while  ((more = isl_stream_yaml_next(s)) > 0) {
		isl_val *val;

		val = isl_stream_read_val(s);
		list = isl_val_list_add(list, val);
	}

	if (more < 0 || isl_stream_yaml_read_end_sequence(s))
		list = isl_val_list_free(list);

	return list;
}

/* Set the (initial) coincident properties of "band" according to
 * the (initial) elements of "coincident".
 */
static __isl_give isl_schedule_band *set_coincident(
	__isl_take isl_schedule_band *band, __isl_take isl_val_list *coincident)
{
	int i;
	int n, m;

	n = isl_schedule_band_n_member(band);
	m = isl_val_list_n_val(coincident);

	for (i = 0; i < n && i < m; ++i) {
		isl_val *v;

		v = isl_val_list_get_val(coincident, i);
		if (!v)
			band = isl_schedule_band_free(band);
		band = isl_schedule_band_member_set_coincident(band, i,
							!isl_val_is_zero(v));
		isl_val_free(v);
	}
	isl_val_list_free(coincident);
	return band;
}

/* Read a subtree with band root node from "s".
 */
static __isl_give isl_schedule_tree *read_band(isl_stream *s)
{
	isl_multi_union_pw_aff *schedule = NULL;
	isl_schedule_tree *tree = NULL;
	isl_val_list *coincident = NULL;
	isl_union_set *options = NULL;
	isl_ctx *ctx;
	isl_schedule_band *band;
	int permutable = 0;
	int more;

	ctx = isl_stream_get_ctx(s);

	do {
		struct isl_token *tok;
		enum isl_schedule_key key;
		char *str;
		isl_val *v;

		key = get_key(s);
		if (isl_stream_yaml_next(s) < 0)
			goto error;

		switch (key) {
		case isl_schedule_key_schedule:
			isl_multi_union_pw_aff_free(schedule);
			tok = isl_stream_next_token(s);
			if (!tok) {
				isl_stream_error(s, NULL, "unexpected EOF");
				goto error;
			}
			str = isl_token_get_str(ctx, tok);
			schedule = isl_multi_union_pw_aff_read_from_str(ctx,
									str);
			free(str);
			isl_token_free(tok);
			if (!schedule)
				goto error;
			break;
		case isl_schedule_key_coincident:
			coincident = read_coincident(s);
			if (!coincident)
				goto error;
			break;
		case isl_schedule_key_permutable:
			v = isl_stream_read_val(s);
			permutable = !isl_val_is_zero(v);
			isl_val_free(v);
			break;
		case isl_schedule_key_options:
			isl_union_set_free(options);
			tok = isl_stream_next_token(s);
			str = isl_token_get_str(ctx, tok);
			options = isl_union_set_read_from_str(ctx, str);
			free(str);
			isl_token_free(tok);
			if (!options)
				goto error;
			break;
		case isl_schedule_key_child:
			isl_schedule_tree_free(tree);
			tree = isl_stream_read_schedule_tree(s);
			if (!tree)
				goto error;
			break;
		default:
			isl_die(ctx, isl_error_invalid, "unexpected key",
				goto error);
		}
	} while ((more = isl_stream_yaml_next(s)) > 0);

	if (more < 0)
		goto error;

	if (!schedule)
		isl_die(ctx, isl_error_invalid, "missing schedule", goto error);

	band = isl_schedule_band_from_multi_union_pw_aff(schedule);
	band = isl_schedule_band_set_permutable(band, permutable);
	if (coincident)
		band = set_coincident(band, coincident);
	if (options)
		band = isl_schedule_band_set_ast_build_options(band, options);
	if (tree)
		tree = isl_schedule_tree_insert_band(tree, band);
	else
		tree = isl_schedule_tree_from_band(band);

	return tree;
error:
	isl_val_list_free(coincident);
	isl_union_set_free(options);
	isl_schedule_tree_free(tree);
	isl_multi_union_pw_aff_free(schedule);
	return NULL;
}

/* Read a subtree with root node of type "type" from "s".
 * The node is represented by a sequence of children.
 */
static __isl_give isl_schedule_tree *read_children(isl_stream *s,
	enum isl_schedule_node_type type)
{
	isl_ctx *ctx;
	isl_schedule_tree_list *list;
	int more;

	ctx = isl_stream_get_ctx(s);

	isl_token_free(isl_stream_next_token(s));

	if (isl_stream_yaml_next(s) < 0)
		return NULL;

	if (isl_stream_yaml_read_start_sequence(s))
		return NULL;

	list = isl_schedule_tree_list_alloc(ctx, 0);
	while ((more = isl_stream_yaml_next(s)) > 0) {
		isl_schedule_tree *tree;

		tree = isl_stream_read_schedule_tree(s);
		list = isl_schedule_tree_list_add(list, tree);
	}

	if (more < 0 || isl_stream_yaml_read_end_sequence(s))
		list = isl_schedule_tree_list_free(list);

	return isl_schedule_tree_from_children(type, list);
}

/* Read a subtree with sequence root node from "s".
 */
static __isl_give isl_schedule_tree *read_sequence(isl_stream *s)
{
	return read_children(s, isl_schedule_node_sequence);
}

/* Read a subtree with set root node from "s".
 */
static __isl_give isl_schedule_tree *read_set(isl_stream *s)
{
	return read_children(s, isl_schedule_node_set);
}

/* Read a schedule (sub)tree from "s".
 *
 * We first determine the type of the root node based on the first
 * mapping key and then hand over to a function tailored to reading
 * nodes of this type.
 */
static __isl_give isl_schedule_tree *isl_stream_read_schedule_tree(
	struct isl_stream *s)
{
	enum isl_schedule_key key;
	struct isl_token *tok;
	isl_schedule_tree *tree = NULL;
	int more;

	if (isl_stream_yaml_read_start_mapping(s))
		return NULL;
	more = isl_stream_yaml_next(s);
	if (more < 0)
		return NULL;
	if (!more) {
		isl_stream_error(s, NULL, "missing key");
		return NULL;
	}

	tok = isl_stream_next_token(s);
	key = extract_key(s, tok);
	isl_stream_push_token(s, tok);
	if (key < 0)
		return NULL;
	switch (key) {
	case isl_schedule_key_context:
		tree = read_context(s);
		break;
	case isl_schedule_key_domain:
		tree = read_domain(s);
		break;
	case isl_schedule_key_contraction:
	case isl_schedule_key_expansion:
		tree = read_expansion(s);
		break;
	case isl_schedule_key_extension:
		tree = read_extension(s);
		break;
	case isl_schedule_key_filter:
		tree = read_filter(s);
		break;
	case isl_schedule_key_guard:
		tree = read_guard(s);
		break;
	case isl_schedule_key_leaf:
		isl_token_free(isl_stream_next_token(s));
		tree = isl_schedule_tree_leaf(isl_stream_get_ctx(s));
		break;
	case isl_schedule_key_mark:
		tree = read_mark(s);
		break;
	case isl_schedule_key_sequence:
		tree = read_sequence(s);
		break;
	case isl_schedule_key_set:
		tree = read_set(s);
		break;
	case isl_schedule_key_schedule:
	case isl_schedule_key_coincident:
	case isl_schedule_key_options:
	case isl_schedule_key_permutable:
		tree = read_band(s);
		break;
	case isl_schedule_key_child:
		isl_die(isl_stream_get_ctx(s), isl_error_unsupported,
			"cannot identity node type", return NULL);
	case isl_schedule_key_error:
		return NULL;
	}

	if (isl_stream_yaml_read_end_mapping(s) < 0) {
		isl_stream_error(s, NULL, "unexpected extra elements");
		return isl_schedule_tree_free(tree);
	}

	return tree;
}

/* Read an isl_schedule from "s".
 */
__isl_give isl_schedule *isl_stream_read_schedule(isl_stream *s)
{
	isl_ctx *ctx;
	isl_schedule_tree *tree;

	if (!s)
		return NULL;

	ctx = isl_stream_get_ctx(s);
	tree = isl_stream_read_schedule_tree(s);
	return isl_schedule_from_schedule_tree(ctx, tree);
}

/* Read an isl_schedule from "input".
 */
__isl_give isl_schedule *isl_schedule_read_from_file(isl_ctx *ctx, FILE *input)
{
	struct isl_stream *s;
	isl_schedule *schedule;

	s = isl_stream_new_file(ctx, input);
	if (!s)
		return NULL;
	schedule = isl_stream_read_schedule(s);
	isl_stream_free(s);

	return schedule;
}

/* Read an isl_schedule from "str".
 */
__isl_give isl_schedule *isl_schedule_read_from_str(isl_ctx *ctx,
	const char *str)
{
	struct isl_stream *s;
	isl_schedule *schedule;

	s = isl_stream_new_str(ctx, str);
	if (!s)
		return NULL;
	schedule = isl_stream_read_schedule(s);
	isl_stream_free(s);

	return schedule;
}
