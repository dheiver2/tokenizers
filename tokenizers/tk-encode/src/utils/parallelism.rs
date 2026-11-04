//! The encode thread pool and its knobs: [`max_threads`] / [`set_max_threads`].

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::OnceLock;
use std::sync::TryLockError;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

/// See [`max_threads`]. 0 means the default.
static MAX_THREADS: AtomicUsize = AtomicUsize::new(0);
/// Bumped in a forked child, which has none of the parent's pool threads.
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
            FORK_GEN.fetch_add(1, Ordering::SeqCst);
        }
        unsafe {
            let _ = libc::pthread_atfork(None, None, Some(child_after_fork));
        }
    });
}

#[cfg(not(unix))]
fn register_fork_handler() {}

/// The cached pool, keyed by what it was built from: a change to either rebuilds it.
struct Slot {
    /// `None` when `max_threads()` is 1: cached too, so a serial process skips the rebuild path.
    pool: Option<Arc<rayon::ThreadPool>>,
    /// [`FORK_GEN`] at build time.
    fork_gen: usize,
    /// `max_threads() - 1` at build time.
    workers: usize,
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

    // Read once and stored as the key: a racing `set_max_threads` is caught by the next call.
    let fork_gen = fork_generation();
    // the caller is one of the `max_threads`
    let workers = max_threads() - 1;
    // no lock: encode serially rather than build a throwaway pool
    let mut guard = lock()?;
    if let Some(slot) = guard.as_ref()
        && slot.fork_gen == fork_gen
        && slot.workers == workers
    {
        return slot.pool.clone();
    }
    // build under the lock: callers that find it held go serial instead of building their own
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
        fork_gen,
        workers,
    });
    drop(guard);

    if let Some(old) = old
        && old.fork_gen != fork_gen
    {
        // built before a fork: dropping it would wait on threads the child doesn't have
        std::mem::forget(old.pool);
    }

    pool
}

/// Max threads one encode uses, caller included: the shared pool holds `max_threads() - 1`, so 1
/// means serial.
///
/// A value set with [`set_max_threads`] always wins, and forked children inherit it. Unset, it is
/// `available_parallelism()` (affinity mask, cgroup quota), except in a forked child, which can't
/// know how many siblings share the cores and runs serially. A fork is only detected if this module
/// was used before it (an encode, `max_threads` or `set_max_threads`).
pub fn max_threads() -> usize {
    register_fork_handler();
    match MAX_THREADS.load(Ordering::Acquire) {
        0 if fork_generation() > 0 => 1,
        0 => available_parallelism(),
        n => n,
    }
}

/// Cached: on Linux it reads the cgroup quota from files, too slow for every encode.
fn available_parallelism() -> usize {
    static CORES: OnceLock<usize> = OnceLock::new();
    *CORES.get_or_init(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
}

/// Set [`max_threads`] for the process. It takes precedence over every default, including the
/// serial one of a forked child, and forked children inherit it. `0` restores the default. Takes
/// effect on the next encode.
pub fn set_max_threads(n: usize) {
    register_fork_handler();
    MAX_THREADS.store(n, Ordering::Release);
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// A forked child defaults to serial; an explicit max still applies.
    #[cfg(unix)]
    #[test]
    fn forked_children_default_to_one_thread() {
        let _serial = POOL_TESTS.lock().unwrap_or_else(|p| p.into_inner());
        set_max_threads(0);
        register_fork_handler();
        // SAFETY: the child only reads atomics and `_exit`s
        let pid = unsafe { libc::fork() };
        if pid == 0 {
            let default = max_threads();
            set_max_threads(3);
            let explicit = max_threads();
            unsafe { libc::_exit(i32::from(!(default == 1 && explicit == 3))) };
        }
        let mut status = 0;
        assert_eq!(unsafe { libc::waitpid(pid, &mut status, 0) }, pid);
        assert!(libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0);
        assert!(max_threads() >= 1, "the parent keeps its default");
    }

    /// Contended callers must reuse the cached pool, not build their own.
    #[test]
    fn concurrent_callers_share_one_pool() {
        let _serial = POOL_TESTS.lock().unwrap_or_else(|p| p.into_inner());
        if max_threads() == 1 {
            return; // no pool on a one-core machine
        }
        // start stale (cached for another size), so the callers race the rebuild too
        pool();
        set_max_threads(max_threads() + 1);
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
        set_max_threads(0);
        assert_eq!(seen.len(), 1, "contended callers built pools of their own");
    }
}
