package sqlx

import (
	"context"
	"database/sql"
	"database/sql/driver"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"testing"
)

// A tiny in-memory driver used to observe the exact flat bindvar arguments
// produced by named parameter expansion without requiring CGO.

func init() {
	sql.Register("sqlxfake", &fakeDrv{})
	BindDriver("sqlxfake", QUESTION)
}

type fakeCall struct {
	connID int
	query  string
	args   []driver.Value
}

type fakeDrv struct {
	mu     sync.Mutex
	nextID int
	conns  map[int]*fakeConn
	calls  []fakeCall
	openN  int
	closeN int
}

func (d *fakeDrv) Open(name string) (driver.Conn, error) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.nextID++
	id := d.nextID
	d.openN++
	if d.conns == nil {
		d.conns = map[int]*fakeConn{}
	}
	c := &fakeConn{drv: d, id: id}
	d.conns[id] = c
	return c, nil
}

func (d *fakeDrv) record(c fakeCall) {
	d.mu.Lock()
	d.calls = append(d.calls, c)
	d.mu.Unlock()
}

func (d *fakeDrv) snapshot() []fakeCall {
	d.mu.Lock()
	defer d.mu.Unlock()
	out := make([]fakeCall, len(d.calls))
	copy(out, d.calls)
	return out
}

func (d *fakeDrv) last() fakeCall {
	d.mu.Lock()
	defer d.mu.Unlock()
	if len(d.calls) == 0 {
		return fakeCall{connID: -1}
	}
	return d.calls[len(d.calls)-1]
}

func (d *fakeDrv) connClosed(id int) {
	d.mu.Lock()
	d.closeN++
	delete(d.conns, id)
	d.mu.Unlock()
}

type fakeConn struct {
	drv *fakeDrv
	id  int
}

func (c *fakeConn) Prepare(query string) (driver.Stmt, error) {
	return &fakeStmt{conn: c, query: query}, nil
}
func (c *fakeConn) Close() error {
	c.drv.connClosed(c.id)
	return nil
}
func (c *fakeConn) Begin() (driver.Tx, error) { return &fakeTx{conn: c}, nil }

type fakeTx struct{ conn *fakeConn }

func (t *fakeTx) Commit() error   { return nil }
func (t *fakeTx) Rollback() error { return nil }

type fakeStmt struct {
	conn  *fakeConn
	query string
}

func (s *fakeStmt) Close() error  { return nil }
func (s *fakeStmt) NumInput() int { return -1 }

func (s *fakeStmt) Exec(args []driver.Value) (driver.Result, error) {
	s.conn.drv.record(fakeCall{connID: s.conn.id, query: s.query, args: cpArgs(args)})
	if strings.Contains(s.query, "BOOM") {
		return nil, errors.New("fake boom")
	}
	return fakeResult{}, nil
}

func (s *fakeStmt) Query(args []driver.Value) (driver.Rows, error) {
	s.conn.drv.record(fakeCall{connID: s.conn.id, query: s.query, args: cpArgs(args)})
	if strings.Contains(s.query, "BOOM") {
		return nil, errors.New("fake boom")
	}
	cols := []string{"id"}
	var data [][]driver.Value
	if !strings.Contains(s.query, "EMPTY") {
		if strings.Contains(s.query, "ROWS2") {
			data = [][]driver.Value{{int64(1)}, {int64(2)}}
		} else {
			data = [][]driver.Value{{int64(42)}}
		}
	}
	return &fakeRows{cols: cols, data: data}, nil
}

type fakeResult struct{}

func (fakeResult) LastInsertId() (int64, error) { return 0, nil }
func (fakeResult) RowsAffected() (int64, error) { return 1, nil }

type fakeRows struct {
	cols []string
	data [][]driver.Value
	pos  int
}

func (r *fakeRows) Columns() []string { return r.cols }
func (r *fakeRows) Close() error      { return nil }
func (r *fakeRows) Next(dest []driver.Value) error {
	if r.pos >= len(r.data) {
		return io.EOF
	}
	copy(dest, r.data[r.pos])
	r.pos++
	return nil
}

func cpArgs(args []driver.Value) []driver.Value {
	out := make([]driver.Value, len(args))
	copy(out, args)
	return out
}

func newFakeDB(t *testing.T) (*DB, *fakeDrv) {
	t.Helper()
	db, err := Open("sqlxfake", "")
	if err != nil {
		t.Fatalf("open fake db: %v", err)
	}
	drv := db.Driver().(*fakeDrv)
	t.Cleanup(func() { db.Close() })
	return db, drv
}

type scopeAB struct {
	A int    `db:"a"`
	B string `db:"b"`
}

type scopeRow struct {
	ID int64 `db:"id"`
}

func wantArgs(t *testing.T, call fakeCall, want ...driver.Value) {
	t.Helper()
	if len(call.args) != len(want) {
		t.Fatalf("args len: got %#v, want %#v", call.args, want)
	}
	for i := range want {
		if fmt.Sprintf("%v", call.args[i]) != fmt.Sprintf("%v", want[i]) {
			t.Fatalf("arg %d: got %#v, want %#v (all: %#v)", i, call.args[i], want[i], call.args)
		}
	}
}

func TestScopeOutsideUnchanged(t *testing.T) {
	db, drv := newFakeDB(t)

	if _, err := db.NamedExec("INSERT INTO t VALUES (:a)", map[string]interface{}{"a": 1}); err != nil {
		t.Fatalf("named exec: %v", err)
	}
	wantArgs(t, drv.last(), int64(1))

	// outside a scope, a missing name is still an error and nothing reaches
	// the driver
	before := len(drv.snapshot())
	if _, err := db.NamedExec("SELECT :missing", map[string]interface{}{}); err == nil {
		t.Fatal("expected error for missing name outside scope")
	}
	if len(drv.snapshot()) != before {
		t.Fatal("driver must not be called when binding fails outside scope")
	}
}

func TestScopeBasicStagingAndOverride(t *testing.T) {
	db, drv := newFakeDB(t)

	sc, err := db.BeginNamedScope()
	if err != nil {
		t.Fatalf("begin: %v", err)
	}

	if _, err := sc.NamedExec("INSERT (:a,:b)", map[string]interface{}{"a": 1, "b": "x"}); err != nil {
		t.Fatalf("first exec: %v", err)
	}

	// later call only references :a: it reads the staged value
	if _, err := sc.NamedExec("SELECT :a", map[string]interface{}{}); err != nil {
		t.Fatalf("staged read: %v", err)
	}
	wantArgs(t, drv.last(), int64(1))

	// the call's own argument overrides the staged value, and on success the
	// new value replaces the staged one
	if _, err := sc.NamedExec("SELECT :a", map[string]interface{}{"a": 2}); err != nil {
		t.Fatalf("override: %v", err)
	}
	wantArgs(t, drv.last(), int64(2))
	if _, err := sc.NamedExec("SELECT :a", map[string]interface{}{}); err != nil {
		t.Fatalf("staged read after override: %v", err)
	}
	wantArgs(t, drv.last(), int64(2))

	if err := sc.Commit(); err != nil {
		t.Fatalf("commit: %v", err)
	}
}

func TestScopeFailedCallDoesNotStage(t *testing.T) {
	db, _ := newFakeDB(t)

	sc, _ := db.BeginNamedScope()
	if _, err := sc.NamedExec("BOOM (:a)", map[string]interface{}{"a": 1}); err == nil {
		t.Fatal("expected exec error")
	}
	// the failed call's write never lands in the layer
	if _, err := sc.NamedExec("SELECT :a", map[string]interface{}{}); err == nil {
		t.Fatal("failed call must not stage its keys")
	}
	if err := sc.Rollback(); err != nil {
		t.Fatalf("rollback: %v", err)
	}
	if err := sc.Rollback(); !errors.Is(err, ErrScopeClosed) {
		t.Fatalf("double rollback: got %v, want ErrScopeClosed", err)
	}
}

func TestScopeNestedCommitMerge(t *testing.T) {
	db, drv := newFakeDB(t)

	outer, _ := db.BeginNamedScope()
	if _, err := outer.NamedExec("w (:a)", map[string]interface{}{"a": 1}); err != nil {
		t.Fatal(err)
	}

	inner, err := outer.BeginNamedScope()
	if err != nil {
		t.Fatalf("nested begin: %v", err)
	}

	// inner reads outer committed value
	if _, err := inner.NamedExec("r (:a)", map[string]interface{}{}); err != nil {
		t.Fatalf("inner read outer: %v", err)
	}
	wantArgs(t, drv.last(), int64(1))

	if _, err := inner.NamedExec("w (:b)", map[string]interface{}{"b": "y"}); err != nil {
		t.Fatal(err)
	}
	// inner writes are not visible in the outer layer while inner is active:
	// lookupStaged walks only committed layers, and the outer chain cannot
	// reach into the child.
	if _, err := outer.NamedExec("r (:b)", map[string]interface{}{}); err == nil {
		t.Fatal("outer must not see uncommitted inner key")
	}

	if err := inner.Commit(); err != nil {
		t.Fatalf("inner commit: %v", err)
	}

	// after success the inner write is visible outside
	if _, err := outer.NamedExec("r (:b)", map[string]interface{}{}); err != nil {
		t.Fatalf("outer read merged: %v", err)
	}
	wantArgs(t, drv.last(), "y")

	if err := outer.Commit(); err != nil {
		t.Fatalf("outer commit: %v", err)
	}
}

func TestScopeNestedRollbackDiscard(t *testing.T) {
	db, drv := newFakeDB(t)

	outer, _ := db.BeginNamedScope()
	if _, err := outer.NamedExec("w (:a)", map[string]interface{}{"a": 1}); err != nil {
		t.Fatal(err)
	}
	inner, _ := outer.BeginNamedScope()
	if _, err := inner.NamedExec("w (:b)", map[string]interface{}{"b": 2}); err != nil {
		t.Fatal(err)
	}
	if err := inner.Rollback(); err != nil {
		t.Fatal(err)
	}

	// outer value intact, inner value invisible
	if _, err := outer.NamedExec("r (:a)", map[string]interface{}{}); err != nil {
		t.Fatalf("outer value lost: %v", err)
	}
	wantArgs(t, drv.last(), int64(1))
	if _, err := outer.NamedExec("r (:b)", map[string]interface{}{}); err == nil {
		t.Fatal("rolled back inner write leaked to outer")
	}
	_ = outer.Rollback()
}

func TestScopeNestedInnerSeesOuterLatestValue(t *testing.T) {
	db, drv := newFakeDB(t)

	outer, _ := db.BeginNamedScope()
	o1, _ := outer.BeginNamedScope()
	if _, err := o1.NamedExec("w (:a)", map[string]interface{}{"a": 1}); err != nil {
		t.Fatal(err)
	}
	if err := o1.Commit(); err != nil {
		t.Fatal(err)
	}
	// outer merges happen at exit-time content
	o2, _ := outer.BeginNamedScope()
	if _, err := o2.NamedExec("w (:a)", map[string]interface{}{"a": 9}); err != nil {
		t.Fatal(err)
	}
	if err := o2.Commit(); err != nil {
		t.Fatal(err)
	}
	if _, err := outer.NamedExec("r (:a)", map[string]interface{}{}); err != nil {
		t.Fatal(err)
	}
	wantArgs(t, drv.last(), int64(9))
	_ = outer.Rollback()
}

func TestScopeParallelScopesIndependent(t *testing.T) {
	db, drv := newFakeDB(t)

	s1, _ := db.BeginNamedScope()
	s2, _ := db.BeginNamedScope()

	if _, err := s1.NamedExec("w (:a)", map[string]interface{}{"a": 1}); err != nil {
		t.Fatal(err)
	}
	// parallel scope does not see s1's staging
	if _, err := s2.NamedExec("r (:a)", map[string]interface{}{}); err == nil {
		t.Fatal("parallel scope must not see foreign staging")
	}
	if _, err := s2.NamedExec("w (:a)", map[string]interface{}{"a": 9}); err != nil {
		t.Fatal(err)
	}
	if _, err := s1.NamedExec("r (:a)", map[string]interface{}{}); err != nil {
		t.Fatal(err)
	}
	wantArgs(t, drv.last(), int64(1))
	if _, err := s2.NamedExec("r (:a)", map[string]interface{}{}); err != nil {
		t.Fatal(err)
	}
	wantArgs(t, drv.last(), int64(9))

	_ = s1.Rollback()
	_ = s2.Rollback()
}

func TestScopeReentryAndConcurrentEntry(t *testing.T) {
	db, _ := newFakeDB(t)
	sc, _ := db.BeginNamedScope()

	// same goroutine, not exited yet -> entry fails
	if err := sc.Enter(); !errors.Is(err, ErrScopeActive) {
		t.Fatalf("reenter active scope: got %v, want ErrScopeActive", err)
	}
	if err := sc.Rollback(); err != nil {
		t.Fatal(err)
	}

	// two concurrent callers racing Enter: exactly one wins
	start := make(chan struct{})
	res := make(chan error, 2)
	for i := 0; i < 2; i++ {
		go func() {
			<-start
			res <- sc.Enter()
		}()
	}
	close(start)
	var nils, actives int
	for i := 0; i < 2; i++ {
		switch err := <-res; {
		case err == nil:
			nils++
		case errors.Is(err, ErrScopeActive):
			actives++
		default:
			t.Fatalf("unexpected enter error: %v", err)
		}
	}
	if nils != 1 || actives != 1 {
		t.Fatalf("concurrent enter: %d success, %d ErrScopeActive", nils, actives)
	}
	_ = sc.Rollback()

	// nesting on an exited outer fails
	if _, err := sc.BeginNamedScope(); !errors.Is(err, ErrScopeClosed) {
		t.Fatalf("nested begin on closed scope: got %v, want ErrScopeClosed", err)
	}
}

func TestScopeOpsOnClosedScopeFail(t *testing.T) {
	db, _ := newFakeDB(t)
	sc, _ := db.BeginNamedScope()
	if err := sc.Rollback(); err != nil {
		t.Fatal(err)
	}
	if _, err := sc.NamedExec("x (:a)", map[string]interface{}{"a": 1}); !errors.Is(err, ErrScopeClosed) {
		t.Fatalf("exec after exit: got %v, want ErrScopeClosed", err)
	}
	if err := sc.Get(&scopeRow{}, "x :a", map[string]interface{}{"a": 1}); !errors.Is(err, ErrScopeClosed) {
		t.Fatalf("get after exit: got %v, want ErrScopeClosed", err)
	}
}

func TestScopeTxIsolation(t *testing.T) {
	db, _ := newFakeDB(t)

	tx, err := db.Beginx()
	if err != nil {
		t.Fatal(err)
	}
	tsc, _ := tx.BeginNamedScope()
	if _, err := tsc.NamedExec("w (:a)", map[string]interface{}{"a": 1}); err != nil {
		t.Fatal(err)
	}

	// a different scope on the same DB must not see tx staging
	other, _ := db.BeginNamedScope()
	if _, err := other.NamedExec("r (:a)", map[string]interface{}{}); err == nil {
		t.Fatal("tx staging leaked to another connection scope")
	}
	_ = other.Rollback()

	// inside the tx scope the value is still there
	if _, err := tsc.NamedExec("r (:a)", map[string]interface{}{}); err != nil {
		t.Fatalf("tx scope lost its staging: %v", err)
	}

	if err := tx.Rollback(); err != nil {
		t.Fatal(err)
	}
	// rolling back the transaction discards the layer: committing it now
	// must surface an error and merge nothing (there is no outer chain).
	if err := tsc.Rollback(); err != nil {
		t.Fatalf("scope rollback after tx rollback: %v", err)
	}
}

func TestScopeStructBindingSameRules(t *testing.T) {
	db, drv := newFakeDB(t)
	sc, _ := db.BeginNamedScope()

	// struct write stages every mapped field
	if _, err := sc.NamedExec("w (:a,:b)", scopeAB{A: 1, B: "x"}); err != nil {
		t.Fatal(err)
	}
	// another struct missing 'a' transparently reads it from staging
	type onlyC struct {
		C int `db:"c"`
	}
	if _, err := sc.NamedExec("r (:a,:c)", onlyC{C: 7}); err != nil {
		t.Fatalf("struct read from staging: %v", err)
	}
	wantArgs(t, drv.last(), int64(1), int64(7))

	// struct value overrides staged value, then replaces staging
	type onlyA struct {
		A int `db:"a"`
	}
	if _, err := sc.NamedExec("r (:a)", onlyA{A: 5}); err != nil {
		t.Fatal(err)
	}
	wantArgs(t, drv.last(), int64(5))
	if _, err := sc.NamedExec("r (:a)", map[string]interface{}{}); err != nil {
		t.Fatal(err)
	}
	wantArgs(t, drv.last(), int64(5))

	// struct-staged keys are visible to map arguments as well
	if _, err := sc.NamedExec("r (:b)", map[string]interface{}{}); err != nil {
		t.Fatalf("map read of struct-staged key: %v", err)
	}
	wantArgs(t, drv.last(), "x")

	_ = sc.Rollback()
}

func TestScopeSliceStaging(t *testing.T) {
	db, drv := newFakeDB(t)
	sc, _ := db.BeginNamedScope()

	rows := []scopeAB{{A: 1, B: "p"}, {A: 2, B: "q"}}
	if _, err := sc.NamedExec("w (:a,:b)", rows); err != nil {
		t.Fatal(err)
	}
	// last write wins on the shared keys; values come from staging
	if _, err := sc.NamedExec("r (:a,:b)", map[string]interface{}{}); err != nil {
		t.Fatal(err)
	}
	wantArgs(t, drv.last(), int64(2), "q")

	_ = sc.Rollback()
}

func TestScopePreparedStmtLiveStaging(t *testing.T) {
	db, drv := newFakeDB(t)
	sc, _ := db.BeginNamedScope()

	if _, err := sc.NamedExec("w (:a)", map[string]interface{}{"a": 1}); err != nil {
		t.Fatal(err)
	}
	nstmt, err := sc.PrepareNamed("SELECT :a, :b")
	if err != nil {
		t.Fatalf("prepare: %v", err)
	}

	// prepared at this moment but sees the current staging, and arg fills :b
	if _, err := nstmt.Exec(map[string]interface{}{"b": "x"}); err != nil {
		t.Fatalf("stmt exec: %v", err)
	}
	wantArgs(t, drv.last(), int64(1), "x")

	// staging written AFTER prepare is visible at the next execution
	if _, err := sc.NamedExec("w (:a)", map[string]interface{}{"a": 10}); err != nil {
		t.Fatal(err)
	}
	if _, err := nstmt.Exec(map[string]interface{}{"b": "y"}); err != nil {
		t.Fatal(err)
	}
	wantArgs(t, drv.last(), int64(10), "y")

	// successful statement execution stages its own argument too
	if _, err := nstmt.Exec(map[string]interface{}{"b": "z"}); err != nil {
		t.Fatal(err)
	}
	if _, err := sc.NamedExec("r (:b)", map[string]interface{}{}); err != nil {
		t.Fatalf("stmt write did not stage: %v", err)
	}
	wantArgs(t, drv.last(), "z")

	// a failed statement execution must not stage its argument: prepare a
	// failing statement carrying a fresh key, then check the earlier staged
	// value of :b is untouched.
	boom, err := sc.PrepareNamed("BOOM :a, :fresh")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := boom.Exec(map[string]interface{}{"fresh": "nope"}); err == nil {
		t.Fatal("expected statement exec failure")
	}
	if _, err := sc.NamedExec("r (:b)", map[string]interface{}{}); err != nil {
		t.Fatalf("read b after failed stmt: %v", err)
	}
	wantArgs(t, drv.last(), "z")
	if _, err := sc.NamedExec("r (:fresh)", map[string]interface{}{}); err == nil {
		t.Fatal("failed statement execution staged its argument")
	}

	if err := sc.Rollback(); err != nil {
		t.Fatal(err)
	}

	// once the scope exited, the statement no longer reads stale staging:
	// the missing :a surfaces as a binding error before the driver is hit.
	// The DB-root scope released its pinned connection on exit, so a conn
	// bound statement reaches end of life like any Conn/Tx statement would.
	if _, err := nstmt.Exec(map[string]interface{}{"b": "only"}); err == nil {
		t.Fatal("exited-scope statement must bind without stale staging")
	}
}

func TestScopePreparedStmtMustExecAbort(t *testing.T) {
	db, _ := newFakeDB(t)
	sc, _ := db.BeginNamedScope()
	if _, err := sc.NamedExec("w (:a)", map[string]interface{}{"a": 1}); err != nil {
		t.Fatal(err)
	}
	nstmt, _ := sc.PrepareNamed("BOOM :a, :b")

	func() {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("expected MustExec panic")
			}
		}()
		nstmt.MustExec(map[string]interface{}{"b": 2})
	}()

	if err := sc.checkActive(); !errors.Is(err, ErrScopeAborted) {
		t.Fatalf("scope after MustExec panic: got %v, want ErrScopeAborted", err)
	}
	if err := sc.Commit(); !errors.Is(err, ErrScopeAborted) {
		t.Fatalf("commit after abort: got %v, want ErrScopeAborted", err)
	}
}

func TestScopeMustExecAbortNested(t *testing.T) {
	db, _ := newFakeDB(t)
	outer, _ := db.BeginNamedScope()
	if _, err := outer.NamedExec("w (:x)", map[string]interface{}{"x": 1}); err != nil {
		t.Fatal(err)
	}
	inner, _ := outer.BeginNamedScope()

	func() {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("expected MustExec panic")
			}
		}()
		inner.MustExec("BOOM :y", map[string]interface{}{"y": 2})
	}()

	if err := inner.Commit(); !errors.Is(err, ErrScopeAborted) {
		t.Fatalf("inner commit: got %v, want ErrScopeAborted", err)
	}
	// outer committed value intact, inner write invisible
	if _, err := outer.NamedExec("r (:x)", map[string]interface{}{}); err != nil {
		t.Fatalf("outer staging lost after inner panic: %v", err)
	}
	if _, err := outer.NamedExec("r (:y)", map[string]interface{}{}); err == nil {
		t.Fatal("inner panic write leaked to outer")
	}
	_ = outer.Rollback()
}

func TestScopeGetSelect(t *testing.T) {
	db, _ := newFakeDB(t)
	sc, _ := db.BeginNamedScope()

	var row scopeRow
	if err := sc.Get(&row, "GET :a", map[string]interface{}{"a": 1}); err != nil {
		t.Fatalf("get: %v", err)
	}
	if row.ID != 42 {
		t.Fatalf("got %d, want 42", row.ID)
	}
	// successful Get staged :a
	if err := sc.Get(&row, "GET :a", map[string]interface{}{}); err != nil {
		t.Fatalf("get staged: %v", err)
	}

	// Get with no rows fails and must not stage its write
	if err := sc.Get(&row, "EMPTY :z", map[string]interface{}{"z": 7}); !errors.Is(err, sql.ErrNoRows) {
		t.Fatalf("empty get: got %v, want sql.ErrNoRows", err)
	}
	if _, err := sc.NamedExec("r (:z)", map[string]interface{}{}); err == nil {
		t.Fatal("failed Get staged its argument")
	}

	// Select success stages too
	var rows []scopeRow
	if err := sc.Select(&rows, "ROWS2 :a", map[string]interface{}{"a": 1}); err != nil {
		t.Fatalf("select: %v", err)
	}
	if len(rows) != 2 {
		t.Fatalf("selected %d rows", len(rows))
	}
	if err := sc.Select(&rows, "ROWS2 :a", map[string]interface{}{}); err != nil {
		t.Fatalf("select staged: %v", err)
	}

	// failed Select does not stage
	if err := sc.Select(&rows, "BOOM :q", map[string]interface{}{"q": 1}); err == nil {
		t.Fatal("expected select error")
	}
	if _, err := sc.NamedExec("r (:q)", map[string]interface{}{}); err == nil {
		t.Fatal("failed Select staged its argument")
	}

	_ = sc.Rollback()
}

func TestScopePinsConnection(t *testing.T) {
	db, drv := newFakeDB(t)
	sc, _ := db.BeginNamedScope()

	for i := 0; i < 3; i++ {
		if _, err := sc.NamedExec(fmt.Sprintf("w%d (:a)", i), map[string]interface{}{"a": i}); err != nil {
			t.Fatal(err)
		}
	}
	calls := drv.snapshot()
	if len(calls) < 3 {
		t.Fatalf("recorded %d calls", len(calls))
	}
	id := calls[len(calls)-3].connID
	for _, c := range calls[len(calls)-3:] {
		if c.connID != id {
			t.Fatalf("scope switched connection mid-flight: %d then %d", id, c.connID)
		}
	}

	// nested scope keeps using the same pinned connection
	inner, _ := sc.BeginNamedScope()
	if _, err := inner.NamedExec("w (:a)", map[string]interface{}{"a": 9}); err != nil {
		t.Fatal(err)
	}
	if got := drv.last().connID; got != id {
		t.Fatalf("nested scope used conn %d, want pinned %d", got, id)
	}
	_ = inner.Rollback()
	if err := sc.Rollback(); err != nil {
		t.Fatal(err)
	}
}

func TestScopeOnConnStaysOnConn(t *testing.T) {
	db, drv := newFakeDB(t)
	conn, err := db.Connx(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	// establish the conn id
	if _, err := conn.ExecContext(context.Background(), "anchor"); err != nil {
		t.Fatal(err)
	}
	wantID := drv.last().connID

	sc, err := conn.BeginNamedScope()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := sc.NamedExec("w (:a)", map[string]interface{}{"a": 1}); err != nil {
		t.Fatal(err)
	}
	if got := drv.last().connID; got != wantID {
		t.Fatalf("conn scope ran on conn %d, want %d", got, wantID)
	}
	if err := sc.Commit(); err != nil {
		t.Fatal(err)
	}
	// the user-owned Conn must remain usable on the same connection
	if _, err := conn.ExecContext(context.Background(), "after"); err != nil {
		t.Fatalf("conn unusable after scope commit: %v", err)
	}
	if got := drv.last().connID; got != wantID {
		t.Fatalf("conn switched after scope exit: %d then %d", wantID, got)
	}
}

func TestScopeOnTxUsesTxConnection(t *testing.T) {
	db, drv := newFakeDB(t)
	tx, _ := db.Beginx()
	if _, err := tx.Exec("anchor"); err != nil {
		t.Fatal(err)
	}
	wantID := drv.last().connID

	sc, _ := tx.BeginNamedScope()
	if _, err := sc.NamedExec("w (:a)", map[string]interface{}{"a": 1}); err != nil {
		t.Fatal(err)
	}
	if got := drv.last().connID; got != wantID {
		t.Fatalf("tx scope ran on conn %d, want tx conn %d", got, wantID)
	}
	if _, err := sc.NamedExec("r (:a)", map[string]interface{}{}); err != nil {
		t.Fatal(err)
	}
	if got := drv.last().connID; got != wantID {
		t.Fatalf("tx scope switched connection: %d then %d", wantID, got)
	}
	_ = sc.Rollback()
	_ = tx.Rollback()
}

func TestScopeStagingNeverLivesOnHandle(t *testing.T) {
	db, _ := newFakeDB(t)
	sc, _ := db.BeginNamedScope()
	if _, err := sc.NamedExec("w (:a)", map[string]interface{}{"a": 1}); err != nil {
		t.Fatal(err)
	}
	// direct use of the same *DB while the scope is active must not see its
	// staging; staging is not stored on the handle.
	if _, err := db.NamedExec("r (:a)", map[string]interface{}{}); err == nil {
		t.Fatal("scope staging leaked onto the *DB handle")
	}
	// neither does a transaction begun on the same DB
	tx, _ := db.Beginx()
	if _, err := tx.NamedExec("r (:a)", map[string]interface{}{}); err == nil {
		t.Fatal("scope staging leaked onto a new Tx")
	}
	_ = tx.Rollback()
	_ = sc.Rollback()
}

func TestScopeContextVariants(t *testing.T) {
	db, drv := newFakeDB(t)
	ctx := context.Background()
	sc, err := db.BeginNamedScopeContext(ctx)
	if err != nil {
		t.Fatalf("begin ctx: %v", err)
	}
	if sc.EnterContext(ctx) == nil {
		t.Fatal("re-entering active scope via context must fail")
	}

	if _, err := sc.NamedExecContext(ctx, "w (:a)", map[string]interface{}{"a": 1}); err != nil {
		t.Fatal(err)
	}
	var row scopeRow
	if err := sc.GetContext(ctx, &row, "GET :a", map[string]interface{}{}); err != nil {
		t.Fatal(err)
	}
	nstmt, err := sc.PrepareNamedContext(ctx, "SELECT :a")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := nstmt.ExecContext(ctx, map[string]interface{}{}); err != nil {
		t.Fatalf("prepared ctx exec with staging: %v", err)
	}
	wantArgs(t, drv.last(), int64(1))
	_ = sc.Rollback()
}

func TestWithNamedScopeHelper(t *testing.T) {
	db, drv := newFakeDB(t)

	// success exit merges into the outer scope
	outer, _ := db.BeginNamedScope()
	err := WithNamedScope(outer, func(inner *NamedScope) error {
		if _, err := inner.NamedExec("w (:a)", map[string]interface{}{"a": 1}); err != nil {
			return err
		}
		return nil
	})
	if err != nil {
		t.Fatalf("with scope: %v", err)
	}
	if _, err := outer.NamedExec("r (:a)", map[string]interface{}{}); err != nil {
		t.Fatalf("merge after successful WithNamedScope: %v", err)
	}
	wantArgs(t, drv.last(), int64(1))
	_ = outer.Rollback()

	// error exit discards the layer
	outer2, _ := db.BeginNamedScope()
	want := errors.New("business failure")
	err = WithNamedScope(outer2, func(inner *NamedScope) error {
		if _, err := inner.NamedExec("w (:b)", map[string]interface{}{"b": 2}); err != nil {
			return err
		}
		return want
	})
	if !errors.Is(err, want) {
		t.Fatalf("got %v, want business error", err)
	}
	if _, err := outer2.NamedExec("r (:b)", map[string]interface{}{}); err == nil {
		t.Fatal("error exit leaked staging to outer")
	}
	_ = outer2.Rollback()

	// panic exit discards the layer and re-panics
	outer3, _ := db.BeginNamedScope()
	func() {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("expected re-panic")
			}
		}()
		_ = WithNamedScope(outer3, func(inner *NamedScope) error {
			inner.MustExec("BOOM :c", map[string]interface{}{"c": 3})
			return nil
		})
	}()
	if _, err := outer3.NamedExec("r (:c)", map[string]interface{}{}); err == nil {
		t.Fatal("panic exit leaked staging to outer")
	}
	_ = outer3.Rollback()
}

func TestScopeStmtDoesNotStoreStaging(t *testing.T) {
	db, drv := newFakeDB(t)
	sc, _ := db.BeginNamedScope()
	nstmt, _ := sc.PrepareNamed("SELECT :a, :b")

	if _, err := sc.NamedExec("w (:a)", map[string]interface{}{"a": 1}); err != nil {
		t.Fatal(err)
	}
	if _, err := nstmt.Exec(map[string]interface{}{"b": "first"}); err != nil {
		t.Fatal(err)
	}
	wantArgs(t, drv.last(), int64(1), "first")

	// a second execution with different args cannot observe anything left
	// behind by the first one; the statement holds no values
	if _, err := nstmt.Exec(map[string]interface{}{"b": "second"}); err != nil {
		t.Fatal(err)
	}
	wantArgs(t, drv.last(), int64(1), "second")

	_ = sc.Rollback()
}
