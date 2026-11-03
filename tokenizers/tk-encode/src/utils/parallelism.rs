//!
//! This module defines helpers to allow optional Rayon usage.
//!

use rayon::iter::IterBridge;
use rayon::prelude::*;
use rayon_cond::CondIterator;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::OnceLock;
use std::sync::TryLockError;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

// Re-export rayon current_num_threads
pub use rayon::current_num_threads;

/// TODO: deprecate
static USED_PARALLELISM: AtomicBool = AtomicBool::new(false);

/// TODO: deprecate
static PARALLELISM: AtomicU8 = AtomicU8::new(0);

/// See [`max_threads`]. 0 means the default.
static MAX_THREADS: AtomicUsize = AtomicUsize::new(0);
/// Counter to track the current version of the pool
/// After forking or changing the number of threads, we need to invalidate and recreate a new pool
/// Old pools will be dropped when they go out of scope (arc refcount goes to 0)
static POOL_GEN: AtomicUsize = AtomicUsize::new(0);

/// Bumped only in a forked child (unlike [`POOL_GEN`], which a resize bumps too).
static FORK_GEN: AtomicUsize = AtomicUsize::new(0);

/// Lets an in-flight encode tell it crossed a fork.
pub(crate) fn fork_generation() -> usize {
    FORK_GEN.load(Ordering::Acquire)
}

/// register an invalidation callback to be called after a fork with pthread_atfork
/// this is required because when forking only the parent thread is copied to the child process so
/// you lose access to the previously built thread pool -> rebuild needed
/// cf the POSIX spec: https://pubs.opengroup.org/onlinepubs/9699919799/functions/fork.html
///
/// `Once`: no caller proceeds before the handler is registered.
#[cfg(unix)]
fn register_fork_handler() {
    use std::sync::Once;
    static REGISTERED: Once = Once::new();
    REGISTERED.call_once(|| {
        unsafe extern "C" fn child_after_fork() {
            POOL_GEN.fetch_add(1, Ordering::SeqCst);
            FORK_GEN.fetch_add(1, Ordering::SeqCst);
        }
        unsafe {
            let _ = libc::pthread_atfork(None, None, Some(child_after_fork));
        }
    });
}

#[cfg(not(unix))]
fn register_fork_handler() {}

#[derive(Clone)]
struct Slot {
    /// `None` when `max_threads()` is 1: cached too, so a serial process skips the rebuild path.
    pool: Option<Arc<rayon::ThreadPool>>,
    version: usize,
    pid: u32,
}

static CELL: Mutex<Option<Slot>> = Mutex::new(None);

type MaybeLockGuard = Option<MutexGuard<'static, Option<Slot>>>;

/// How many times [`lock`] retries a held lock before giving up.
const LOCK_ATTEMPTS: u32 = 64;

/// Take the pool lock, briefly waiting out a holder (it only guards an `Arc` clone or swap).
/// Bounded: a child forked while the lock was held inherits it locked forever.
fn lock() -> MaybeLockGuard {
    for attempt in 0..LOCK_ATTEMPTS {
        match CELL.try_lock() {
            Ok(g) => return Some(g),
            Err(TryLockError::Poisoned(p)) => return Some(p.into_inner()),
            Err(TryLockError::WouldBlock) if attempt < LOCK_ATTEMPTS / 2 => std::hint::spin_loop(),
            Err(TryLockError::WouldBlock) => std::thread::yield_now(),
        }
    }
    None
}

pub(crate) fn pool() -> Option<Arc<rayon::ThreadPool>> {
    register_fork_handler();

    let generation = POOL_GEN.load(Ordering::Acquire);
    // no lock: encode serially rather than build a throwaway pool
    let mut guard = lock()?;
    if let Some(slot) = guard.as_ref()
        && generation == slot.version
    {
        return slot.pool.clone();
    }
    // build under the lock: callers that find it held go serial instead of building their own

    // the caller is one of the `max_threads`
    let workers = max_threads() - 1;
    let pool = if workers == 0 {
        None
    } else {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .thread_name(|i| format!("tk-encode-{i}"))
            .build()
            .ok()?;
        Some(Arc::new(pool))
    };
    let old = guard.replace(Slot {
        pool: pool.clone(),
        version: generation,
        pid: std::process::id(),
    });
    drop(guard);

    if let Some(old) = old
        && old.pid != std::process::id()
    {
        // mem::forget is here to avoid deadlocking on the pool drop after forking
        std::mem::forget(old.pool);
    }

    pool
}

/// Max threads per encode, caller included: the shared pool holds `max_threads() - 1`, so 1 means
/// serial. Defaults to `available_parallelism()` (affinity mask, cgroup quota).
pub fn max_threads() -> usize {
    match MAX_THREADS.load(Ordering::Acquire) {
        0 => available_parallelism(),
        n => n,
    }
}

/// Cached: on Linux it reads the cgroup quota from files, too slow for every encode.
fn available_parallelism() -> usize {
    static CORES: OnceLock<usize> = OnceLock::new();
    *CORES.get_or_init(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
}

fn invalidate() {
    POOL_GEN.fetch_add(1, Ordering::SeqCst);
}

/// Set [`max_threads`] process-wide; `0` restores the default. Takes effect on the next encode.
pub fn set_max_threads(n: usize) {
    MAX_THREADS.store(n, Ordering::Release);
    invalidate();
}

/// Whether [`set_parallelism`] was called.
pub fn is_parallelism_configured() -> bool {
    get_override_parallelism().is_some()
}

/// Check if at some point we used a parallel iterator
pub fn has_parallelism_been_used() -> bool {
    USED_PARALLELISM.load(Ordering::SeqCst)
}

/// Get internally set parallelism
fn get_override_parallelism() -> Option<bool> {
    match PARALLELISM.load(Ordering::SeqCst) {
        0 => None,
        1 => Some(false),
        2 => Some(true),
        _ => unreachable!(),
    }
}

/// Whether the `MaybeParallel*` iterators run in parallel: [`set_parallelism`], else `true`.
pub fn get_parallelism() -> bool {
    get_override_parallelism().unwrap_or(true)
}

/// Turn the `MaybeParallel*` iterators on or off for the whole process.
pub fn set_parallelism(val: bool) {
    PARALLELISM.store(if val { 2 } else { 1 }, Ordering::SeqCst);
}

/// Allows to convert into an iterator that can be executed either parallelly or serially,
/// according to [`get_parallelism`].
pub trait MaybeParallelIterator<P, S>
where
    P: ParallelIterator,
    S: Iterator<Item = P::Item>,
{
    /// Convert ourself in a CondIterator, that will be executed either in parallel or serially,
    /// based solely on [`get_parallelism`]
    fn into_maybe_par_iter(self) -> CondIterator<P, S>;
    /// Convert ourself in a CondIterator, that will be executed either in parallel or serially,
    /// based on both [`get_parallelism`] and the provided bool.
    /// Both must be true to run with parallelism activated.
    fn into_maybe_par_iter_cond(self, cond: bool) -> CondIterator<P, S>;
}

impl<P, S, I> MaybeParallelIterator<P, S> for I
where
    I: IntoParallelIterator<Iter = P, Item = P::Item> + IntoIterator<IntoIter = S, Item = S::Item>,
    P: ParallelIterator,
    S: Iterator<Item = P::Item>,
{
    fn into_maybe_par_iter(self) -> CondIterator<P, S> {
        let parallelism = get_parallelism();
        if parallelism {
            USED_PARALLELISM.store(true, Ordering::SeqCst);
        }
        CondIterator::new(self, parallelism)
    }

    fn into_maybe_par_iter_cond(self, cond: bool) -> CondIterator<P, S> {
        if cond {
            self.into_maybe_par_iter()
        } else {
            CondIterator::from_serial(self)
        }
    }
}

/// Shared reference version of MaybeParallelIterator, works the same but returns an iterator
/// over references, does not consume self
pub trait MaybeParallelRefIterator<'data, P, S>
where
    P: ParallelIterator,
    S: Iterator<Item = P::Item>,
    P::Item: 'data,
{
    fn maybe_par_iter(&'data self) -> CondIterator<P, S>;
    fn maybe_par_iter_cond(&'data self, cond: bool) -> CondIterator<P, S>;
}

impl<'data, P, S, I: 'data + ?Sized> MaybeParallelRefIterator<'data, P, S> for I
where
    &'data I: MaybeParallelIterator<P, S>,
    P: ParallelIterator,
    S: Iterator<Item = P::Item>,
    P::Item: 'data,
{
    fn maybe_par_iter(&'data self) -> CondIterator<P, S> {
        self.into_maybe_par_iter()
    }

    fn maybe_par_iter_cond(&'data self, cond: bool) -> CondIterator<P, S> {
        self.into_maybe_par_iter_cond(cond)
    }
}

/// Exclusive reference version of MaybeParallelIterator, works the same but returns an iterator
/// over mutable references, does not consume self
pub trait MaybeParallelRefMutIterator<'data, P, S>
where
    P: ParallelIterator,
    S: Iterator<Item = P::Item>,
    P::Item: 'data,
{
    fn maybe_par_iter_mut(&'data mut self) -> CondIterator<P, S>;
    fn maybe_par_iter_mut_cond(&'data mut self, cond: bool) -> CondIterator<P, S>;
}

impl<'data, P, S, I: 'data + ?Sized> MaybeParallelRefMutIterator<'data, P, S> for I
where
    &'data mut I: MaybeParallelIterator<P, S>,
    P: ParallelIterator,
    S: Iterator<Item = P::Item>,
    P::Item: 'data,
{
    fn maybe_par_iter_mut(&'data mut self) -> CondIterator<P, S> {
        self.into_maybe_par_iter()
    }

    fn maybe_par_iter_mut_cond(&'data mut self, cond: bool) -> CondIterator<P, S> {
        self.into_maybe_par_iter_cond(cond)
    }
}

/// Converts any serial iterator into a CondIterator, that can either run parallelly or serially.
pub trait MaybeParallelBridge<T, S>
where
    S: Iterator<Item = T> + Send,
    T: Send,
{
    fn maybe_par_bridge(self) -> CondIterator<IterBridge<S>, S>;
    fn maybe_par_bridge_cond(self, cond: bool) -> CondIterator<IterBridge<S>, S>;
}

impl<T, S> MaybeParallelBridge<T, S> for S
where
    S: Iterator<Item = T> + Send,
    T: Send,
{
    fn maybe_par_bridge(self) -> CondIterator<IterBridge<S>, S> {
        let iter = CondIterator::from_serial(self);

        if get_parallelism() {
            USED_PARALLELISM.store(true, Ordering::SeqCst);
            CondIterator::from_parallel(iter.into_parallel().right().unwrap())
        } else {
            iter
        }
    }

    fn maybe_par_bridge_cond(self, cond: bool) -> CondIterator<IterBridge<S>, S> {
        if cond {
            self.maybe_par_bridge()
        } else {
            CondIterator::from_serial(self)
        }
    }
}

/// Allows to convert into `chunks` that can be executed either parallelly or serially.
pub trait MaybeParallelSlice<'data, T>
where
    T: Sync,
{
    /// Create a CondIterator, that will be executed either in parallel or serially,
    /// based solely on [`get_parallelism`]
    fn maybe_par_chunks(
        &'_ self,
        chunk_size: usize,
    ) -> CondIterator<rayon::slice::Chunks<'_, T>, std::slice::Chunks<'_, T>>;
    /// Create a CondIterator, that will be executed either in parallel or serially,
    /// based on both [`get_parallelism`] and the provided bool.
    /// Both must be true to run with parallelism activated.
    fn maybe_par_chunks_cond(
        &'_ self,
        cond: bool,
        chunk_size: usize,
    ) -> CondIterator<rayon::slice::Chunks<'_, T>, std::slice::Chunks<'_, T>>;
}

impl<T> MaybeParallelSlice<'_, T> for [T]
where
    T: Sync,
{
    fn maybe_par_chunks(
        &'_ self,
        chunk_size: usize,
    ) -> CondIterator<rayon::slice::Chunks<'_, T>, std::slice::Chunks<'_, T>> {
        let parallelism = get_parallelism();
        if parallelism {
            CondIterator::from_parallel(self.par_chunks(chunk_size))
        } else {
            CondIterator::from_serial(self.chunks(chunk_size))
        }
    }
    fn maybe_par_chunks_cond(
        &'_ self,
        cond: bool,
        chunk_size: usize,
    ) -> CondIterator<rayon::slice::Chunks<'_, T>, std::slice::Chunks<'_, T>> {
        if cond {
            self.maybe_par_chunks(chunk_size)
        } else {
            CondIterator::from_serial(self.chunks(chunk_size))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maybe_parallel_iterator() {
        let mut v = vec![1u32, 2, 3, 4, 5, 6];

        assert_eq!(v.maybe_par_iter().sum::<u32>(), 21);
        assert_eq!(
            v.maybe_par_iter_mut()
                .map(|v| {
                    *v *= 2;
                    *v
                })
                .sum::<u32>(),
            42
        );
        assert_eq!(v.maybe_par_iter().sum::<u32>(), 42);
        assert_eq!(v.into_maybe_par_iter().sum::<u32>(), 42);
    }

    /// Tests touching the process-wide pool run one at a time: a resize reads as a rebuild.
    static POOL_TESTS: Mutex<()> = Mutex::new(());

    /// The caller counts: the pool holds `max - 1` threads, none for a max of 1.
    #[test]
    fn max_threads_counts_the_caller() {
        let _serial = POOL_TESTS.lock().unwrap_or_else(|p| p.into_inner());
        set_max_threads(3);
        assert_eq!(max_threads(), 3);
        assert_eq!(pool().expect("a pool").current_num_threads(), 2);
        set_max_threads(1);
        assert!(pool().is_none(), "one thread is the caller alone");
        set_max_threads(0);
        assert_eq!(
            max_threads(),
            std::thread::available_parallelism().map_or(1, |n| n.get())
        );
    }

    /// Contended callers must reuse the cached pool, not build their own.
    #[test]
    fn concurrent_callers_share_one_pool() {
        let _serial = POOL_TESTS.lock().unwrap_or_else(|p| p.into_inner());
        if max_threads() == 1 {
            return; // no pool on a one-core machine
        }
        // start stale, so the callers race the rebuild too
        invalidate();
        let callers: Vec<_> = (0..32)
            .map(|_| {
                std::thread::spawn(|| {
                    (0..2_000)
                        .filter_map(|_| pool())
                        .map(|pool| Arc::as_ptr(&pool) as usize)
                        .collect::<std::collections::HashSet<_>>()
                })
            })
            .collect();
        let seen: std::collections::HashSet<_> = callers
            .into_iter()
            .flat_map(|caller| caller.join().unwrap())
            .collect();
        assert_eq!(seen.len(), 1, "contended callers built pools of their own");
    }

    #[test]
    fn test_maybe_parallel_slice() {
        let v = [1, 2, 3, 4, 5];

        let chunks: Vec<_> = v.maybe_par_chunks(2).collect();
        assert_eq!(chunks, vec![&[1, 2][..], &[3, 4], &[5]]);
    }
}
