package sqlx

// Named Query Support
//
//  * BindMap - bind query bindvars to map/struct args
//	* NamedExec, NamedQuery - named query w/ struct or map
//  * NamedStmt - a pre-compiled named query which is a prepared statement
//
// Internal Interfaces:
//
//  * compileNamedQuery - rebind a named query, returning a query and list of names
//  * bindArgs, bindMapArgs, bindAnyArgs - given a list of names, return an arglist
//
import (
	"bytes"
	"database/sql"
	"errors"
	"fmt"
	"reflect"
	"regexp"
	"strconv"
	"unicode"

	"github.com/jmoiron/sqlx/reflectx"
)

// NamedStmt is a prepared statement that executes named queries.  Prepare it
// how you would execute a NamedQuery, but pass in a struct or map when executing.
type NamedStmt struct {
	Params      []string
	QueryString string
	Stmt        *Stmt
	// scope, when non-nil, references the live named parameter staging scope
	// in which this statement was prepared.  It is consulted on every
	// Exec/Query; no staged value is ever copied into the statement, so
	// concurrent executions can neither freeze a snapshot nor pollute each
	// other.  nil for statements prepared outside of a scope.
	scope namedStmtScope
}

// namedStager is implemented by an active named parameter staging scope.
// lookupStaged resolves a name that the current call's own argument does not
// carry, walking the innermost layer first and then committed outer layers.
type namedStager interface {
	lookupStaged(name string) (interface{}, bool)
}

// namedStmtScope is the part of a NamedScope used by NamedStmt while binding,
// executing and staging the writes of a prepared named statement.
type namedStmtScope interface {
	namedStager
	// active reports whether the scope is currently entered and usable.
	active() bool
	// stageArg merges the names carried by a successful call's argument
	// (map, struct or slice thereof) into the scope's current layer.
	stageArg(arg interface{})
	// abort discards every write staged in the current layer; it is invoked
	// on the Must* panic paths before the panic is re-raised.
	abort()
}

// Close closes the named statement.
func (n *NamedStmt) Close() error {
	return n.Stmt.Close()
}

// scopeInUse returns the live scope the statement must bind against, or nil
// when the statement was prepared outside of a scope (or its scope has been
// exited, in which case plain binding semantics apply again).
func (n *NamedStmt) scopeInUse() namedStmtScope {
	if n.scope != nil && n.scope.active() {
		return n.scope
	}
	return nil
}

// bindStmtArgs expands the prepared parameter names against arg, overlaying
// the live staging chain when the statement runs inside a scope.
func (n *NamedStmt) bindStmtArgs(arg interface{}, sc namedStmtScope) ([]interface{}, error) {
	if sc == nil {
		return bindAnyArgs(n.Params, arg, n.Stmt.Mapper)
	}
	return bindAnyArgsStaged(n.Params, arg, n.Stmt.Mapper, sc)
}

// Exec executes a named statement using the struct passed.
// Any named placeholder parameters are replaced with fields from arg.
func (n *NamedStmt) Exec(arg interface{}) (sql.Result, error) {
	sc := n.scopeInUse()
	args, err := n.bindStmtArgs(arg, sc)
	if err != nil {
		return *new(sql.Result), err
	}
	res, err := n.Stmt.Exec(args...)
	if err != nil {
		return *new(sql.Result), err
	}
	if sc != nil {
		sc.stageArg(arg)
	}
	return res, nil
}

// Query executes a named statement using the struct argument, returning rows.
// Any named placeholder parameters are replaced with fields from arg.
func (n *NamedStmt) Query(arg interface{}) (*sql.Rows, error) {
	sc := n.scopeInUse()
	args, err := n.bindStmtArgs(arg, sc)
	if err != nil {
		return nil, err
	}
	rows, err := n.Stmt.Query(args...)
	if err != nil {
		return nil, err
	}
	if sc != nil {
		sc.stageArg(arg)
	}
	return rows, nil
}

// QueryRow executes a named statement against the database.  Because sqlx cannot
// create a *sql.Row with an error condition pre-set for binding errors, sqlx
// returns a *sqlx.Row instead.
// Any named placeholder parameters are replaced with fields from arg.
func (n *NamedStmt) QueryRow(arg interface{}) *Row {
	sc := n.scopeInUse()
	args, err := n.bindStmtArgs(arg, sc)
	if err != nil {
		return &Row{err: err}
	}
	if sc != nil {
		sc.stageArg(arg)
	}
	return n.Stmt.QueryRowx(args...)
}

// MustExec execs a NamedStmt, panicing on error
// Any named placeholder parameters are replaced with fields from arg.
func (n *NamedStmt) MustExec(arg interface{}) sql.Result {
	res, err := n.Exec(arg)
	if err != nil {
		if sc := n.scope; sc != nil {
			sc.abort()
		}
		panic(err)
	}
	return res
}

// Queryx using this NamedStmt
// Any named placeholder parameters are replaced with fields from arg.
func (n *NamedStmt) Queryx(arg interface{}) (*Rows, error) {
	r, err := n.Query(arg)
	if err != nil {
		return nil, err
	}
	return &Rows{Rows: r, Mapper: n.Stmt.Mapper, unsafe: isUnsafe(n)}, err
}

// QueryRowx this NamedStmt.  Because of limitations with QueryRow, this is
// an alias for QueryRow.
// Any named placeholder parameters are replaced with fields from arg.
func (n *NamedStmt) QueryRowx(arg interface{}) *Row {
	return n.QueryRow(arg)
}

// Select using this NamedStmt
// Any named placeholder parameters are replaced with fields from arg.
func (n *NamedStmt) Select(dest interface{}, arg interface{}) error {
	rows, err := n.Queryx(arg)
	if err != nil {
		return err
	}
	// if something happens here, we want to make sure the rows are Closed
	defer rows.Close()
	return scanAll(rows, dest, false)
}

// Get using this NamedStmt
// Any named placeholder parameters are replaced with fields from arg.
func (n *NamedStmt) Get(dest interface{}, arg interface{}) error {
	r := n.QueryRowx(arg)
	return r.scanAny(dest, false)
}

// Unsafe creates an unsafe version of the NamedStmt
func (n *NamedStmt) Unsafe() *NamedStmt {
	r := &NamedStmt{Params: n.Params, Stmt: n.Stmt, QueryString: n.QueryString, scope: n.scope}
	r.Stmt.unsafe = true
	return r
}

// A union interface of preparer and binder, required to be able to prepare
// named statements (as the bindtype must be determined).
type namedPreparer interface {
	Preparer
	binder
}

func prepareNamed(p namedPreparer, query string) (*NamedStmt, error) {
	bindType := BindType(p.DriverName())
	q, args, err := compileNamedQuery([]byte(query), bindType)
	if err != nil {
		return nil, err
	}
	stmt, err := Preparex(p, q)
	if err != nil {
		return nil, err
	}
	return &NamedStmt{
		QueryString: q,
		Params:      args,
		Stmt:        stmt,
	}, nil
}

// convertMapStringInterface attempts to convert v to map[string]interface{}.
// Unlike v.(map[string]interface{}), this function works on named types that
// are convertible to map[string]interface{} as well.
func convertMapStringInterface(v interface{}) (map[string]interface{}, bool) {
	var m map[string]interface{}
	mtype := reflect.TypeOf(m)
	t := reflect.TypeOf(v)
	if !t.ConvertibleTo(mtype) {
		return nil, false
	}
	return reflect.ValueOf(v).Convert(mtype).Interface().(map[string]interface{}), true

}

func bindAnyArgs(names []string, arg interface{}, m *reflectx.Mapper) ([]interface{}, error) {
	return bindAnyArgsStaged(names, arg, m, nil)
}

// bindAnyArgsStaged is bindAnyArgs with an optional staging layer: names not
// carried by the call's own argument are resolved through the live scope
// chain.  Map and struct arguments take the identical code path.
func bindAnyArgsStaged(names []string, arg interface{}, m *reflectx.Mapper, st namedStager) ([]interface{}, error) {
	if maparg, ok := convertMapStringInterface(arg); ok {
		return bindMapArgsStaged(names, maparg, st)
	}
	return bindArgsStaged(names, arg, m, st)
}

// private interface to generate a list of interfaces from a given struct
// type, given a list of names to pull out of the struct.  Used by public
// BindStruct interface.
func bindArgs(names []string, arg interface{}, m *reflectx.Mapper) ([]interface{}, error) {
	return bindArgsStaged(names, arg, m, nil)
}

func bindArgsStaged(names []string, arg interface{}, m *reflectx.Mapper, st namedStager) ([]interface{}, error) {
	arglist := make([]interface{}, 0, len(names))

	// grab the indirected value of arg
	var v reflect.Value
	for v = reflect.ValueOf(arg); v.Kind() == reflect.Ptr; {
		v = v.Elem()
	}

	err := m.TraversalsByNameFunc(v.Type(), names, func(i int, t []int) error {
		if len(t) == 0 {
			// the argument itself does not carry this name; fall back to the
			// staging chain before reporting it missing
			if st != nil {
				if val, ok := st.lookupStaged(names[i]); ok {
					arglist = append(arglist, val)
					return nil
				}
			}
			return fmt.Errorf("could not find name %s in %#v", names[i], arg)
		}

		val := reflectx.FieldByIndexesReadOnly(v, t)
		arglist = append(arglist, val.Interface())

		return nil
	})

	return arglist, err
}

// like bindArgs, but for maps.
func bindMapArgs(names []string, arg map[string]interface{}) ([]interface{}, error) {
	return bindMapArgsStaged(names, arg, nil)
}

func bindMapArgsStaged(names []string, arg map[string]interface{}, st namedStager) ([]interface{}, error) {
	arglist := make([]interface{}, 0, len(names))

	for _, name := range names {
		if val, ok := arg[name]; ok {
			arglist = append(arglist, val)
			continue
		}
		// a key the call did not carry may be supplied by a committed layer
		// of the staging chain
		if st != nil {
			if val, ok := st.lookupStaged(name); ok {
				arglist = append(arglist, val)
				continue
			}
		}
		return arglist, fmt.Errorf("could not find name %s in %#v", name, arg)
	}
	return arglist, nil
}

// bindStruct binds a named parameter query with fields from a struct argument.
// The rules for binding field names to parameter names follow the same
// conventions as for StructScan, including obeying the `db` struct tags.
func bindStruct(bindType int, query string, arg interface{}, m *reflectx.Mapper) (string, []interface{}, error) {
	return bindStructStaged(bindType, query, arg, m, nil)
}

func bindStructStaged(bindType int, query string, arg interface{}, m *reflectx.Mapper, st namedStager) (string, []interface{}, error) {
	bound, names, err := compileNamedQuery([]byte(query), bindType)
	if err != nil {
		return "", []interface{}{}, err
	}

	arglist, err := bindAnyArgsStaged(names, arg, m, st)
	if err != nil {
		return "", []interface{}{}, err
	}

	return bound, arglist, nil
}

var valuesReg = regexp.MustCompile(`\)\s*(?i)VALUES\s*\(`)

func findMatchingClosingBracketIndex(s string) int {
	count := 0
	for i, ch := range s {
		if ch == '(' {
			count++
		}
		if ch == ')' {
			count--
			if count == 0 {
				return i
			}
		}
	}
	return 0
}

func fixBound(bound string, loop int) string {
	loc := valuesReg.FindStringIndex(bound)
	// defensive guard when "VALUES (...)" not found
	if len(loc) < 2 {
		return bound
	}

	openingBracketIndex := loc[1] - 1
	index := findMatchingClosingBracketIndex(bound[openingBracketIndex:])
	// defensive guard. must have closing bracket
	if index == 0 {
		return bound
	}
	closingBracketIndex := openingBracketIndex + index + 1

	var buffer bytes.Buffer

	buffer.WriteString(bound[0:closingBracketIndex])
	for i := 0; i < loop-1; i++ {
		buffer.WriteString(",")
		buffer.WriteString(bound[openingBracketIndex:closingBracketIndex])
	}
	buffer.WriteString(bound[closingBracketIndex:])
	return buffer.String()
}

// bindArray binds a named parameter query with fields from an array or slice of
// structs argument.
func bindArray(bindType int, query string, arg interface{}, m *reflectx.Mapper) (string, []interface{}, error) {
	return bindArrayStaged(bindType, query, arg, m, nil)
}

func bindArrayStaged(bindType int, query string, arg interface{}, m *reflectx.Mapper, st namedStager) (string, []interface{}, error) {
	// do the initial binding with QUESTION;  if bindType is not question,
	// we can rebind it at the end.
	bound, names, err := compileNamedQuery([]byte(query), QUESTION)
	if err != nil {
		return "", []interface{}{}, err
	}
	arrayValue := reflect.ValueOf(arg)
	arrayLen := arrayValue.Len()
	if arrayLen == 0 {
		return "", []interface{}{}, fmt.Errorf("length of array is 0: %#v", arg)
	}
	var arglist = make([]interface{}, 0, len(names)*arrayLen)
	for i := 0; i < arrayLen; i++ {
		elemArglist, err := bindAnyArgsStaged(names, arrayValue.Index(i).Interface(), m, st)
		if err != nil {
			return "", []interface{}{}, err
		}
		arglist = append(arglist, elemArglist...)
	}
	if arrayLen > 1 {
		bound = fixBound(bound, arrayLen)
	}
	// adjust binding type if we weren't on question
	if bindType != QUESTION {
		bound = Rebind(bindType, bound)
	}
	return bound, arglist, nil
}

// bindMap binds a named parameter query with a map of arguments.
func bindMap(bindType int, query string, args map[string]interface{}) (string, []interface{}, error) {
	return bindMapStaged(bindType, query, args, nil)
}

func bindMapStaged(bindType int, query string, args map[string]interface{}, st namedStager) (string, []interface{}, error) {
	bound, names, err := compileNamedQuery([]byte(query), bindType)
	if err != nil {
		return "", []interface{}{}, err
	}

	arglist, err := bindMapArgsStaged(names, args, st)
	return bound, arglist, err
}

// -- Compilation of Named Queries

// Allow digits and letters in bind params;  additionally runes are
// checked against underscores, meaning that bind params can have be
// alphanumeric with underscores.  Mind the difference between unicode
// digits and numbers, where '5' is a digit but '五' is not.
var allowedBindRunes = []*unicode.RangeTable{unicode.Letter, unicode.Digit}

// FIXME: this function isn't safe for unicode named params, as a failing test
// can testify.  This is not a regression but a failure of the original code
// as well.  It should be modified to range over runes in a string rather than
// bytes, even though this is less convenient and slower.  Hopefully the
// addition of the prepared NamedStmt (which will only do this once) will make
// up for the slightly slower ad-hoc NamedExec/NamedQuery.

// compile a NamedQuery into an unbound query (using the '?' bindvar) and
// a list of names.
func compileNamedQuery(qs []byte, bindType int) (query string, names []string, err error) {
	names = make([]string, 0, 10)
	rebound := make([]byte, 0, len(qs))

	inName := false
	last := len(qs) - 1
	currentVar := 1
	name := make([]byte, 0, 10)

	for i, b := range qs {
		// a ':' while we're in a name is an error
		if b == ':' {
			// if this is the second ':' in a '::' escape sequence, append a ':'
			if inName && i > 0 && qs[i-1] == ':' {
				rebound = append(rebound, ':')
				inName = false
				continue
			} else if inName {
				err = errors.New("unexpected `:` while reading named param at " + strconv.Itoa(i))
				return query, names, err
			}
			inName = true
			name = []byte{}
		} else if inName && i > 0 && b == '=' && len(name) == 0 {
			rebound = append(rebound, ':', '=')
			inName = false
			continue
			// if we're in a name, and this is an allowed character, continue
		} else if inName && (unicode.IsOneOf(allowedBindRunes, rune(b)) || b == '_' || b == '.') && i != last {
			// append the byte to the name if we are in a name and not on the last byte
			name = append(name, b)
			// if we're in a name and it's not an allowed character, the name is done
		} else if inName {
			inName = false
			// if this is the final byte of the string and it is part of the name, then
			// make sure to add it to the name
			if i == last && unicode.IsOneOf(allowedBindRunes, rune(b)) {
				name = append(name, b)
			}
			// add the string representation to the names list
			names = append(names, string(name))
			// add a proper bindvar for the bindType
			switch bindType {
			// oracle only supports named type bind vars even for positional
			case NAMED:
				rebound = append(rebound, ':')
				rebound = append(rebound, name...)
			case QUESTION, UNKNOWN:
				rebound = append(rebound, '?')
			case DOLLAR:
				rebound = append(rebound, '$')
				for _, b := range strconv.Itoa(currentVar) {
					rebound = append(rebound, byte(b))
				}
				currentVar++
			case AT:
				rebound = append(rebound, '@', 'p')
				for _, b := range strconv.Itoa(currentVar) {
					rebound = append(rebound, byte(b))
				}
				currentVar++
			}
			// add this byte to string unless it was not part of the name
			if i != last {
				rebound = append(rebound, b)
			} else if !unicode.IsOneOf(allowedBindRunes, rune(b)) {
				rebound = append(rebound, b)
			}
		} else {
			// this is a normal byte and should just go onto the rebound query
			rebound = append(rebound, b)
		}
	}

	return string(rebound), names, err
}

// BindNamed binds a struct or a map to a query with named parameters.
// DEPRECATED: use sqlx.Named` instead of this, it may be removed in future.
func BindNamed(bindType int, query string, arg interface{}) (string, []interface{}, error) {
	return bindNamedMapper(bindType, query, arg, mapper())
}

// Named takes a query using named parameters and an argument and
// returns a new query with a list of args that can be executed by
// a database.  The return value uses the `?` bindvar.
func Named(query string, arg interface{}) (string, []interface{}, error) {
	return bindNamedMapper(QUESTION, query, arg, mapper())
}

func bindNamedMapper(bindType int, query string, arg interface{}, m *reflectx.Mapper) (string, []interface{}, error) {
	return bindNamedMapperStaged(bindType, query, arg, m, nil)
}

// bindNamedMapperStaged is bindNamedMapper with an optional staging layer
// consulted for names the call's own argument does not carry.
func bindNamedMapperStaged(bindType int, query string, arg interface{}, m *reflectx.Mapper, st namedStager) (string, []interface{}, error) {
	t := reflect.TypeOf(arg)
	k := t.Kind()
	switch {
	case k == reflect.Map && t.Key().Kind() == reflect.String:
		m, ok := convertMapStringInterface(arg)
		if !ok {
			return "", nil, fmt.Errorf("sqlx.bindNamedMapper: unsupported map type: %T", arg)
		}
		return bindMapStaged(bindType, query, m, st)
	case k == reflect.Array || k == reflect.Slice:
		return bindArrayStaged(bindType, query, arg, m, st)
	default:
		return bindStructStaged(bindType, query, arg, m, st)
	}
}

// NamedQuery binds a named query and then runs Query on the result using the
// provided Ext (sqlx.Tx, sqlx.Db).  It works with both structs and with
// map[string]interface{} types.
func NamedQuery(e Ext, query string, arg interface{}) (*Rows, error) {
	q, args, err := bindNamedMapper(BindType(e.DriverName()), query, arg, mapperFor(e))
	if err != nil {
		return nil, err
	}
	return e.Queryx(q, args...)
}

// NamedExec uses BindStruct to get a query executable by the driver and
// then runs Exec on the result.  Returns an error from the binding
// or the query execution itself.
func NamedExec(e Ext, query string, arg interface{}) (sql.Result, error) {
	q, args, err := bindNamedMapper(BindType(e.DriverName()), query, arg, mapperFor(e))
	if err != nil {
		return nil, err
	}
	return e.Exec(q, args...)
}
