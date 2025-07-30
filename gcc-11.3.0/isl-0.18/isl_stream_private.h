#include <isl_int.h>
#include <isl/stream.h>
#include <isl_yaml.h>

struct isl_token {
	int type;

	unsigned int on_new_line : 1;
	unsigned is_keyword : 1;
	int line;
	int col;

	union {
		isl_int	v;
		char	*s;
		isl_map *map;
		isl_pw_aff *pwaff;
	} u;
};

struct isl_token *isl_token_new(isl_ctx *ctx,
	int line, int col, unsigned on_new_line);

/* An input stream that may be either a file or a string.
 *
 * line and col are the line and column number of the next character (1-based).
 * start_line and start_col are set by isl_stream_getc to point
 * to the position of the returned character.
 * last_line is the line number of the previous token.
 *
 * yaml_state and yaml_indent keep track of the currently active YAML
 * elements.  yaml_size is the size of these arrays, while yaml_depth
 * is the number of elements currently in use.
 * yaml_state and yaml_indent may be NULL if no YAML parsing is being
 * performed.
 * yaml_state keeps track of what is expected next at each level.
 * yaml_indent keeps track of the indentation at each level, with
 * ISL_YAML_INDENT_FLOW meaning that the element is in flow format
 * (such that the indentation is not relevant).
 */
struct isl_stream {
	struct isl_ctx	*ctx;
	FILE        	*file;
	const char  	*str;
	int	    	line;
	int	    	col;
	int		start_line;
	int		start_col;
	int		last_line;
	int	    	eof;

	char	    	*buffer;
	size_t	    	size;
	size_t	    	len;
	int	    	c;
	int		un[5];
	int		n_un;

	struct isl_token	*tokens[5];
	int	    	n_token;

	struct isl_hash_table	*keywords;
	enum isl_token_type	 next_type;

	int			yaml_depth;
	int			yaml_size;
	enum isl_yaml_state	*yaml_state;
	int			*yaml_indent;
};
