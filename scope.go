package sqlx

// Explicit named parameter staging scopes.
//
// A NamedScope overlays a transactional staging layer on top of the existing
// named parameter expansion path without replacing the underlying DB, Conn,
// Tx or Stmt handles.  Inside a scope, every successful Named* call stages
// the names carried by its argument: expansions read the argument first, then
// the current layer, then committed outer layers.  A successful Commit merges
// the layer into its parent; Rollback, an error exit or a Must* panic discards
// the layer and leaves every outer value untouched.

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"reflect"
	"sync"

	"github.com/jmoiron/sqlx/reflectx"
)

// Sentinel errors returned by named parameter staging scopes.
var (
	// ErrScopeActive is returned when entering a NamedScope that has not been
	// exited yet, whether by the same goroutine or by a concurrent caller.
	ErrScopeActive = errors.New("sqlx: named scope is already active")

	// ErrScopeClosed is returned by operations on a NamedScope that has not
	// been entered, or that has already been committed or rolled back.
	ErrScopeClosed = errors.New("sqlx: named scope is not active")

	// ErrScopeAborted is returned by Commit when the scope was aborted (for
	// example by a panicking MustExec) instead of exited successfully; all
	// writes of the aborted layer have been discarded.
	ErrScopeAborted = errors.New("sqlx: named scope was aborted and its staged writes were discarded")
)

// scopeExec is the fixed execution handle a scope chain runs on.  *sql.DB,
// *sql.Tx and *sql.Conn all satisfy it.
type scopeExec interface {
	ExecContext(ctx context.Context, query string, args ...interface{}) (sql.Result, error)
	QueryContext(ctx context.Context, query string, args ...interface{}) (*sql.Rows, error)
	QueryRowContext(ctx context.Context, query string, args ...interface{}) *sql.Row
	PrepareContext(ctx context.Context, query string) (*sql.Stmt, error)
}

// NamedScope is an explicitly entered named parameter staging scope.  It must
// be created with BeginNamedScope (on DB, Conn, Tx or on another scope) and
// exited exactly once with Commit or Rollback.
//
// A scope never replaces the handle it runs on: scopes entered on a Tx keep
// using that exact transaction, scopes entered on a Conn keep using that exact
// connection, and scopes entered on a DB pin a single connection for the
// lifetime of the scope so staging can never follow a mid-scope connection
// switch.  Staged values only ever live in the scope chain itself, so nothing
// leaks to other Conns or Txs of the same DB.
//
// A NamedScope must not be entered by two calls concurrently.
type NamedScope struct {
	// lifeMu guards the active/aborted lifecycle and handle pinning.
	lifeMu sync.Mutex
	// mu guards the values map of this layer.
	mu sync.RWMutex

	// parent is the immediately outer scope; nil for a root scope.
	parent *NamedScope

	// The immutable execution handle, set at construction.  Exactly one of
	// db/tx/conn/pinned is used by target().
	db   *DB
	tx   *Tx
	conn *Conn
	// pinned is the connection a DB root pins for the scope's lifetime.
	// Children share their parent's pinned connection without owning it.
	pinned *sql.Conn
	// pinnedOwn reports whether this scope must release pinned on exit.
	pinnedOwn bool

	driverName string
	mapper     *reflectx.Mapper
	unsafe     bool

	entered bool
	aborted bool
	// values holds the writes successfully staged in this layer.  It is
	// never shared with statement objects or other scopes.
	values map[string]interface{}
}

// BeginNamedScope enters a named parameter staging scope on this DB.  The
// scope executes every statement on one pinned connection of the pool; the
// connection is released when the scope is committed or rolled back.
func (db *DB) BeginNamedScope() (*NamedScope, error) {
	return db.BeginNamedScopeContext(context.Background())
}

// BeginNamedScopeContext is BeginNamedScope with the provided context used to
// acquire the pinned connection.
func (db *DB) BeginNamedScopeContext(ctx context.Context) (*NamedScope, error) {
	s := &NamedScope{
		db:         db,
		driverName: db.driverName,
		mapper:     db.Mapper,
		unsafe:     db.unsafe,
	}
	return s, s.EnterContext(ctx)
}

// BeginNamedScope enters a named parameter staging scope bound to this
// transaction.  The scope uses the exact Tx it was entered on; staged writes
// never leak to other Conns or Txs of the originating DB.
func (tx *Tx) BeginNamedScope() (*NamedScope, error) {
	return tx.BeginNamedScopeContext(context.Background())
}

// BeginNamedScopeContext is BeginNamedScope for a Tx; the context only governs
// entry (a Tx is already bound to one connection).
func (tx *Tx) BeginNamedScopeContext(ctx context.Context) (*NamedScope, error) {
	s := &NamedScope{
		tx:         tx,
		driverName: tx.driverName,
		mapper:     tx.Mapper,
		unsafe:     tx.unsafe,
	}
	return s, s.EnterContext(ctx)
}

// BeginNamedScope enters a named parameter staging scope bound to this
// connection.  The scope can never continue on another connection.
func (c *Conn) BeginNamedScope() (*NamedScope, error) {
	return c.BeginNamedScopeContext(context.Background())
}

// BeginNamedScopeContext is BeginNamedScope for a Conn.
func (c *Conn) BeginNamedScopeContext(ctx context.Context) (*NamedScope, error) {
	s := &NamedScope{
		conn:       c,
		driverName: c.driverName,
		mapper:     c.Mapper,
		unsafe:     c.unsafe,
	}
	return s, s.EnterContext(ctx)
}

// BeginNamedScope enters a nested scope inside s.  The inner scope executes on
// the exact same handle as the outer one and reads committed outer values for
// keys it does not cover; its own writes only become visible to s after the
// inner scope exits with Commit.  Entering a nested scope fails while s is not
// active.
func (s *NamedScope) BeginNamedScope() (*NamedScope, error) {
	return s.BeginNamedScopeContext(context.Background())
}

// BeginNamedScopeContext is BeginNamedScope for a nested scope; the context
// governs acquisition if the outer scope ever needs to (re)pin a connection.
func (s *NamedScope) BeginNamedScopeContext(ctx context.Context) (*NamedScope, error) {
	s.lifeMu.Lock()
	if !s.entered {
		s.lifeMu.Unlock()
		return nil, ErrScopeClosed
	}
	child := &NamedScope{
		parent:     s,
		db:         s.db,
		tx:         s.tx,
		conn:       s.conn,
		pinned:     s.pinned,
		pinnedOwn:  false,
		driverName: s.driverName,
		mapper:     s.mapper,
		unsafe:     s.unsafe,
	}
	s.lifeMu.Unlock()
	return child, child.EnterContext(ctx)
}

// Enter activates a scope that is not currently active.  It fails with
// ErrScopeActive if the scope is already active, including re-entrant entry by
// the same goroutine, and concurrent entry therefore also fails.  It fails
// with ErrScopeClosed if an outer scope is no longer active.
func (s *NamedScope) Enter() error {
	return s.EnterContext(context.Background())
}

// EnterContext is Enter with a context governing connection pinning.
func (s *NamedScope) EnterContext(ctx context.Context) error {
	s.lifeMu.Lock()
	defer s.lifeMu.Unlock()

	if s.entered {
		return ErrScopeActive
	}
	if s.parent != nil && !s.parent.isActiveLocked() {
		return ErrScopeClosed
	}
	if s.db != nil && s.pinned == nil {
		conn, err := s.db.DB.Conn(ctx)
		if err != nil {
			return err
		}
		s.pinned = conn
		s.pinnedOwn = true
	}
	s.entered = true
	s.aborted = false
	return nil
}

// Commit exits the scope successfully.  The writes staged in this layer, taken
// at exit time, are merged into the parent scope; a root scope simply ends.
// After Commit the scope is inactive and its staging is gone.  It returns
// ErrScopeClosed if the scope was already exited, or ErrScopeAborted if the
// scope was aborted by a Must* panic.
func (s *NamedScope) Commit() error {
	merged, err := s.exit(true)
	if err != nil {
		return err
	}
	if s.parent != nil && !s.parent.mergeActive(merged) {
		return ErrScopeClosed
	}
	return nil
}

// Rollback exits the scope discarding every write staged in this layer;
// committed outer values are left untouched.  It returns ErrScopeClosed if the
// scope was already exited.
func (s *NamedScope) Rollback() error {
	_, err := s.exit(false)
	return err
}

func (s *NamedScope) exit(commit bool) (map[string]interface{}, error) {
	s.lifeMu.Lock()
	if !s.entered {
		s.lifeMu.Unlock()
		return nil, ErrScopeClosed
	}
	s.entered = false
	aborted := s.aborted

	var merged map[string]interface{}
	if commit && !aborted {
		s.mu.RLock()
		if len(s.values) > 0 {
			merged = make(map[string]interface{}, len(s.values))
			for k, v := range s.values {
				merged[k] = v
			}
		}
		s.mu.RUnlock()
	}

	var pinned *sql.Conn
	if s.pinnedOwn {
		pinned = s.pinned
		s.pinned = nil
		s.pinnedOwn = false
	}
	s.mu.Lock()
	s.values = nil
	s.mu.Unlock()
	s.lifeMu.Unlock()

	if pinned != nil {
		if cerr := pinned.Close(); cerr != nil {
			return merged, cerr
		}
	}
	if commit && aborted {
		return nil, ErrScopeAborted
	}
	return merged, nil
}

func (s *NamedScope) isActiveLocked() bool {
	return s.entered
}

func (s *NamedScope) active() bool {
	s.lifeMu.Lock()
	defer s.lifeMu.Unlock()
	return s.entered && !s.aborted
}

func (s *NamedScope) checkActive() error {
	s.lifeMu.Lock()
	defer s.lifeMu.Unlock()
	if !s.entered {
		return ErrScopeClosed
	}
	if s.aborted {
		return ErrScopeAborted
	}
	return nil
}

// abort throws away all writes of the current layer and marks it aborted.
// Used by Must* methods on their panic path.
func (s *NamedScope) abort() {
	s.lifeMu.Lock()
	defer s.lifeMu.Unlock()
	s.mu.Lock()
	s.aborted = true
	s.values = nil
	s.mu.Unlock()
}

// mergeActive merges a successfully exited child layer into s while s itself
// is still active; returns false (without merging) if s was already exited.
func (s *NamedScope) mergeActive(m map[string]interface{}) bool {
	s.lifeMu.Lock()
	defer s.lifeMu.Unlock()
	if !s.entered || s.aborted {
		return false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.values == nil && len(m) > 0 {
		s.values = make(map[string]interface{}, len(m))
	}
	for k, v := range m {
		s.values[k] = v
	}
	return true
}

// target returns the fixed handle every scope statement executes on.
func (s *NamedScope) target() scopeExec {
	switch {
	case s.pinned != nil:
		return s.pinned
	case s.tx != nil:
		return s.tx.Tx
	case s.conn != nil:
		return s.conn.Conn
	default:
		return s.db.DB
	}
}

// lookupStaged resolves a name from the staging chain, innermost layer first.
func (s *NamedScope) lookupStaged(name string) (interface{}, bool) {
	for sc := s; sc != nil; sc = sc.parent {
		sc.mu.RLock()
		v, ok := sc.values[name]
		sc.mu.RUnlock()
		if ok {
			return v, true
		}
	}
	return nil, false
}

// stageArg records the names carried by a successful call's argument into the
// current layer.  Map and struct arguments follow identical rules; slices and
// arrays contribute the union of their elements' names, last write winning.
func (s *NamedScope) stageArg(arg interface{}) {
	values := stagedValues(arg, s.mapper)
	if len(values) == 0 {
		return
	}
	s.lifeMu.Lock()
	defer s.lifeMu.Unlock()
	if !s.entered || s.aborted {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.values == nil {
		s.values = make(map[string]interface{}, len(values))
	}
	for k, v := range values {
		s.values[k] = v
	}
}

// stagedValues extracts the named key/value pairs carried by an argument.
func stagedValues(arg interface{}, m *reflectx.Mapper) map[string]interface{} {
	if arg == nil {
		return nil
	}
	if maparg, ok := convertMapStringInterface(arg); ok {
		return maparg
	}

	v := reflect.ValueOf(arg)
	for v.Kind() == reflect.Ptr {
		v = v.Elem()
	}

	switch v.Kind() {
	case reflect.Struct:
		tm := m.TypeMap(v.Type())
		out := make(map[string]interface{}, len(tm.Names))
		for name, fi := range tm.Names {
			if fv, ok := readableField(v, fi.Index); ok {
				out[name] = fv.Interface()
			}
		}
		return out
	case reflect.Slice, reflect.Array:
		var out map[string]interface{}
		for i := 0; i < v.Len(); i++ {
			elem := stagedValues(v.Index(i).Interface(), m)
			for k, val := range elem {
				if out == nil {
					out = make(map[string]interface{}, len(elem))
				}
				out[k] = val
			}
		}
		return out
	default:
		return nil
	}
}

// readableField follows a struct traversal for staging without allocating nil
// pointers; fields unreachable through nil pointers are skipped.
func readableField(v reflect.Value, index []int) (reflect.Value, bool) {
	for _, i := range index {
		v = reflect.Indirect(v)
		if !v.IsValid() || v.Kind() != reflect.Struct {
			return reflect.Value{}, false
		}
		v = v.Field(i)
		switch v.Kind() {
		case reflect.Ptr, reflect.Interface:
			if v.IsNil() {
				return reflect.Value{}, false
			}
		}
	}
	if !v.IsValid() || !v.CanInterface() {
		return reflect.Value{}, false
	}
	return v, true
}

// bind expands a named query against the argument and the live staging chain.
func (s *NamedScope) bind(query string, arg interface{}) (string, []interface{}, error) {
	return bindNamedMapperStaged(BindType(s.driverName), query, arg, s.mapper, s)
}

func (s *NamedScope) queryRowxContext(ctx context.Context, query string, args []interface{}) *Row {
	rows, err := s.target().QueryContext(ctx, query, args...)
	return &Row{rows: rows, err: err, unsafe: s.unsafe, Mapper: s.mapper}
}

// NamedExecContext binds a named query against the argument and the staging
// chain and executes it on the scope's fixed handle.  On success the names
// carried by arg are staged in the current layer; on error nothing is staged.
func (s *NamedScope) NamedExecContext(ctx context.Context, query string, arg interface{}) (sql.Result, error) {
	if err := s.checkActive(); err != nil {
		return nil, err
	}
	q, args, err := s.bind(query, arg)
	if err != nil {
		return nil, err
	}
	res, err := s.target().ExecContext(ctx, q, args...)
	if err != nil {
		return nil, err
	}
	s.stageArg(arg)
	return res, nil
}

// NamedQueryContext binds a named query against the argument and the staging
// chain and runs it on the scope's fixed handle.  When the query is dispatched
// successfully the names carried by arg are staged in the current layer.
func (s *NamedScope) NamedQueryContext(ctx context.Context, query string, arg interface{}) (*Rows, error) {
	if err := s.checkActive(); err != nil {
		return nil, err
	}
	q, args, err := s.bind(query, arg)
	if err != nil {
		return nil, err
	}
	rows, err := s.target().QueryContext(ctx, q, args...)
	if err != nil {
		return nil, err
	}
	s.stageArg(arg)
	return &Rows{Rows: rows, unsafe: s.unsafe, Mapper: s.mapper}, nil
}

// GetContext binds a named query against the argument and the staging chain,
// runs it and scans the single row into dest.  The names carried by arg are
// only staged when the query and scan succeed.
func (s *NamedScope) GetContext(ctx context.Context, dest interface{}, query string, arg interface{}) error {
	if err := s.checkActive(); err != nil {
		return err
	}
	q, args, err := s.bind(query, arg)
	if err != nil {
		return err
	}
	r := s.queryRowxContext(ctx, q, args)
	if err := r.scanAny(dest, false); err != nil {
		return err
	}
	s.stageArg(arg)
	return nil
}

// SelectContext binds a named query against the argument and the staging
// chain, runs it and scans all rows into dest.  The names carried by arg are
// only staged when the query and scan succeed.
func (s *NamedScope) SelectContext(ctx context.Context, dest interface{}, query string, arg interface{}) error {
	if err := s.checkActive(); err != nil {
		return err
	}
	q, args, err := s.bind(query, arg)
	if err != nil {
		return err
	}
	rows, err := s.target().QueryContext(ctx, q, args...)
	if err != nil {
		return err
	}
	r := &Rows{Rows: rows, unsafe: s.unsafe, Mapper: s.mapper}
	defer r.Close()
	if err := scanAll(r, dest, false); err != nil {
		return err
	}
	s.stageArg(arg)
	return nil
}

// MustExecContext runs NamedExecContext and panics on any binding or
// execution error.  On the panic path every write of the current layer is
// discarded before the panic is re-raised.
func (s *NamedScope) MustExecContext(ctx context.Context, query string, arg interface{}) sql.Result {
	res, err := s.NamedExecContext(ctx, query, arg)
	if err != nil {
		s.abort()
		panic(err)
	}
	return res
}

// PrepareNamedContext compiles and prepares a named query on the scope's fixed
// handle.  The returned statement binds against the live staging chain at
// every Exec/Query call rather than against a snapshot taken at prepare time;
// no staged value is ever stored on the statement.
func (s *NamedScope) PrepareNamedContext(ctx context.Context, query string) (*NamedStmt, error) {
	if err := s.checkActive(); err != nil {
		return nil, err
	}
	bound, names, err := compileNamedQuery([]byte(query), BindType(s.driverName))
	if err != nil {
		return nil, err
	}
	stmt, err := s.target().PrepareContext(ctx, bound)
	if err != nil {
		return nil, err
	}
	return &NamedStmt{
		QueryString: bound,
		Params:      names,
		Stmt:        &Stmt{Stmt: stmt, unsafe: s.unsafe, Mapper: s.mapper},
		scope:       s,
	}, nil
}

// NamedExec is NamedExecContext with context.Background.
func (s *NamedScope) NamedExec(query string, arg interface{}) (sql.Result, error) {
	return s.NamedExecContext(context.Background(), query, arg)
}

// NamedQuery is NamedQueryContext with context.Background.
func (s *NamedScope) NamedQuery(query string, arg interface{}) (*Rows, error) {
	return s.NamedQueryContext(context.Background(), query, arg)
}

// Get is GetContext with context.Background.
func (s *NamedScope) Get(dest interface{}, query string, arg interface{}) error {
	return s.GetContext(context.Background(), dest, query, arg)
}

// Select is SelectContext with context.Background.
func (s *NamedScope) Select(dest interface{}, query string, arg interface{}) error {
	return s.SelectContext(context.Background(), dest, query, arg)
}

// MustExec is MustExecContext with context.Background.
func (s *NamedScope) MustExec(query string, arg interface{}) sql.Result {
	return s.MustExecContext(context.Background(), query, arg)
}

// PrepareNamed is PrepareNamedContext with context.Background.
func (s *NamedScope) PrepareNamed(query string) (*NamedStmt, error) {
	return s.PrepareNamedContext(context.Background(), query)
}

// WithNamedScope enters a staging scope on e (which must be *DB, *Conn, *Tx or
// an outer *NamedScope), runs fn inside it and exits it when fn returns.  A
// nil return from fn commits the layer (merging it into the outer scope); a
// non-nil return rolls it back, so the error is returned and no staged write
// becomes visible outside the scope.  A panic (including a Must* panic) also
// discards the layer before the panic is re-raised.
func WithNamedScope(e interface{}, fn func(*NamedScope) error) (err error) {
	var s *NamedScope
	switch h := e.(type) {
	case *DB:
		s, err = h.BeginNamedScope()
	case *Conn:
		s, err = h.BeginNamedScope()
	case *Tx:
		s, err = h.BeginNamedScope()
	case *NamedScope:
		s, err = h.BeginNamedScope()
	default:
		return fmt.Errorf("sqlx: WithNamedScope: unsupported scope entry handle %T", e)
	}
	if err != nil {
		return err
	}

	defer func() {
		if p := recover(); p != nil {
			_ = s.Rollback()
			panic(p)
		}
		if err != nil {
			_ = s.Rollback()
			return
		}
		err = s.Commit()
	}()

	err = fn(s)
	return err
}
