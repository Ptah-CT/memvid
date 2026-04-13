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

mod frame_jsonl;
mod split_memvid;
mod mv2x;

pub use frame_jsonl::{FrameJsonl, write_frame_jsonl, read_all_frames};
pub use split_memvid::SplitMemvid;
