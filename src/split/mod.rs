//! Split-file backend for Memvid: `.mv2d` (JSONL data) + `.mv2x` (binary indices).
//!
//! This module provides [`SplitMemvid`], an alternative to the single-file [`Memvid`]
//! that separates plaintext data from binary indices:
//!
//! - **`.mv2d`** — Append-only JSONL, one line per frame. Human-readable, grep-able, git-diffable.
//! - **`.mv2x`** — Binary indices (Lex/Vec/Time/LogicMesh/Offsets). Derived artifact, rebuildable.
//!
//! # Design Principles
//!
//! 1. Source of truth is `.mv2d` (plaintext JSONL)
//! 2. `.mv2x` is a derived, deletable, rebuildable cache
//! 3. Writes to `.mv2d` hold a flock() for microseconds (no embedding under lock)
//! 4. Index updates happen asynchronously after the data write
//!
//! # Example
//!
//! ```no_run
//! use memvid_core::split::{SplitMemvid, FrameJsonl};
//!
//! let mut store = SplitMemvid::open("agent_memory.mv2d").unwrap();
//!
//! // Write frames (µs, append-only)
//! store.put("Agent learned something new").unwrap();
//! store.put_frame(
//!     FrameJsonl::new(0, "SurrealDB drops after 32min idle")
//!         .with_tag("sector", "episodic")
//!         .with_tag("agent", "anubis")
//! ).unwrap();
//!
//! // Build search indices (async in production)
//! store.rebuild_lex_index().unwrap();
//! store.flush_index().unwrap();
//!
//! // Search
//! let hits = store.lex_search("SurrealDB idle", 5);
//! ```

pub mod frame_jsonl;
pub mod mv2x;
pub mod split_memvid;

pub use frame_jsonl::{FrameJsonl, FrameStatus};
pub use mv2x::Mv2xIndex;
pub use split_memvid::{CompactResult, SplitMemvid};
