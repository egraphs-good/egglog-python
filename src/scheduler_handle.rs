use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use egglog::scheduler::{FreshScheduler, Matches, Scheduler, SchedulerId};
use log::{debug, info};
use pyo3::prelude::*;

static NEXT_SCHEDULER_OWNER_ID: AtomicUsize = AtomicUsize::new(1);

pub(crate) fn next_scheduler_owner_id() -> usize {
    NEXT_SCHEDULER_OWNER_ID.fetch_add(1, Ordering::Relaxed)
}

#[pyclass(frozen)]
#[derive(Clone)]
pub struct SchedulerHandle {
    pub(crate) owner_id: usize,
    pub(crate) scheduler_id: SchedulerId,
}

impl SchedulerHandle {
    pub(crate) fn new(owner_id: usize, scheduler_id: SchedulerId) -> Self {
        Self {
            owner_id,
            scheduler_id,
        }
    }
}

#[pymethods]
impl SchedulerHandle {
    fn __repr__(&self) -> String {
        "SchedulerHandle(...)".to_string()
    }
}

pub(crate) fn backlog_backoff(
    match_limit: usize,
    ban_length: usize,
    haskell_backoff: bool,
) -> Box<dyn Scheduler> {
    Box::new(BackOffScheduler {
        state: BackOffState::new(match_limit, ban_length, haskell_backoff),
    })
}

pub(crate) fn fresh_backoff(
    match_limit: usize,
    ban_length: usize,
    haskell_backoff: bool,
) -> Box<dyn FreshScheduler> {
    Box::new(BackOffEggScheduler {
        state: BackOffState::new(match_limit, ban_length, haskell_backoff),
    })
}

#[derive(Debug, Clone)]
struct BackOffState {
    default_match_limit: usize,
    default_ban_length: usize,
    haskell_backoff: bool,
    stats: HashMap<String, RuleStats>,
}

#[derive(Debug, Clone)]
struct RuleStats {
    iteration: usize,
    times_applied: usize,
    banned_until: usize,
    times_banned: usize,
    match_limit: usize,
    ban_length: usize,
}

impl BackOffState {
    fn new(default_match_limit: usize, default_ban_length: usize, haskell_backoff: bool) -> Self {
        Self {
            default_match_limit,
            default_ban_length,
            haskell_backoff,
            stats: HashMap::new(),
        }
    }

    fn get_stats(&mut self, rule: String) -> &mut RuleStats {
        self.stats.entry(rule).or_insert_with(|| RuleStats {
            times_applied: 0,
            banned_until: 0,
            times_banned: 0,
            match_limit: self.default_match_limit,
            ban_length: self.default_ban_length,
            iteration: 0,
        })
    }

    fn can_stop(&mut self, rules: &[&str]) -> bool {
        let stats = &mut self.stats;
        let n_stats = stats.len();

        let mut banned: Vec<(&str, RuleStats)> = rules
            .iter()
            .filter_map(|rule| {
                let s = stats.remove(*rule).unwrap();
                if s.banned_until > s.iteration {
                    Some((*rule, s))
                } else {
                    None
                }
            })
            .collect();

        let result = if banned.is_empty() {
            true
        } else {
            let min_delta = banned
                .iter()
                .map(|(_, s)| {
                    assert!(s.banned_until >= s.iteration);
                    s.banned_until - s.iteration
                })
                .min()
                .expect("banned cannot be empty here");

            let mut unbanned = vec![];
            for (name, s) in &mut banned {
                s.banned_until -= min_delta;
                if s.banned_until == s.iteration {
                    unbanned.push(*name);
                }
            }

            assert!(!unbanned.is_empty());
            info!(
                "Banned {}/{}, fast-forwarded by {} to unban {}",
                banned.len(),
                n_stats,
                min_delta,
                unbanned.join(", "),
            );

            false
        };

        for (rule, s) in banned {
            stats.insert(rule.to_owned(), s);
        }

        result
    }

    fn should_search(&mut self, rule: &str) -> bool {
        let stats = self.get_stats(rule.to_owned());
        stats.iteration += 1;

        if stats.iteration < stats.banned_until {
            debug!(
                "Skipping {} ({}-{}), banned until {}...",
                rule, stats.times_applied, stats.times_banned, stats.banned_until,
            );
            false
        } else {
            true
        }
    }

    fn choose_or_ban(&mut self, rule: &str, matches: &mut Matches) -> bool {
        let haskell_backoff = self.haskell_backoff;
        let stats = self.get_stats(rule.to_owned());
        let threshold = stats
            .match_limit
            .checked_shl(stats.times_banned as u32)
            .unwrap();
        // Haskell's backoff scheduler counts substitution width, not just the
        // number of matched tuples.
        let total_len = if haskell_backoff {
            matches.match_size().saturating_mul(matches.tuple_len())
        } else {
            matches.match_size()
        };
        if total_len > threshold {
            let ban_length = stats.ban_length << stats.times_banned;
            stats.times_banned += 1;
            stats.banned_until = stats.iteration + ban_length;
            info!(
                "Banning {} ({}-{}) for {} iters: {} < {}",
                rule, stats.times_applied, stats.times_banned, ban_length, threshold, total_len,
            );
            false
        } else {
            stats.times_applied += 1;
            debug!(
                "Choosing all matches for {} ({}-{})",
                rule, stats.times_applied, stats.times_banned
            );
            matches.choose_all();
            true
        }
    }
}

#[derive(Debug, Clone)]
struct BackOffScheduler {
    state: BackOffState,
}

impl Scheduler for BackOffScheduler {
    fn can_stop(&mut self, rules: &[&str], _ruleset: &str) -> bool {
        self.state.can_stop(rules)
    }

    fn filter_matches(&mut self, rule: &str, _ruleset: &str, matches: &mut Matches) -> bool {
        self.state.should_search(rule) && self.state.choose_or_ban(rule, matches)
    }
}

#[derive(Debug, Clone)]
struct BackOffEggScheduler {
    state: BackOffState,
}

impl FreshScheduler for BackOffEggScheduler {
    fn should_search(&mut self, rule: &str, _ruleset: &str) -> bool {
        self.state.should_search(rule)
    }

    fn can_stop(&mut self, rules: &[&str], _ruleset: &str) -> bool {
        self.state.can_stop(rules)
    }

    fn filter_matches(&mut self, rule: &str, _ruleset: &str, matches: &mut Matches) {
        let _ = self.state.choose_or_ban(rule, matches);
    }
}
