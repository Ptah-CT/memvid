//! [`SplitMemvid`] — split-file Memvid backend.
//!
//! Manages a `.mv2d` / `.mv2x` pair. Data writes go to JSONL,
//! index updates are tracked separately.

use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::Result;
use super::frame_jsonl::{self, FrameJsonl, FrameStatus};
use super::mv2x::{Mv2xIndex, Mv2xHeader};

/// A split-file Memvid store managing `.mv2d` (data) + `.mv2x` (indices).
pub struct SplitMemvid {
    /// Path to the `.mv2d` file (JSONL data).
    mv2d_path: PathBuf,
    /// Path to the `.mv2x` file (binary indices).
    mv2x_path: PathBuf,
    /// Open file handle for `.mv2d` (append mode).
    mv2d_file: File,
    /// In-memory index state.
    index: Mv2xIndex,
    /// Next frame_id to assign.
    next_frame_id: u64,
    /// Set of tombstoned/superseded frame_ids (for active-frame filtering).
    inactive_frames: HashSet<u64>,
}

impl SplitMemvid {
    /// Create a new split store, or open an existing one.
    ///
    /// If the `.mv2d` file exists, loads existing frames and reconciles the index.
    /// If not, creates both files empty.
    pub fn open(mv2d_path: impl AsRef<Path>) -> Result<Self> {
        let mv2d_path = mv2d_path.as_ref().to_path_buf();
        let mv2x_path = mv2d_path.with_extension("mv2x");

        let mv2d_file = frame_jsonl::open_mv2d(&mv2d_path)?;

        // Determine next frame_id from existing data
        let next_frame_id = match frame_jsonl::last_frame_id(&mv2d_path)? {
            Some(id) => id + 1,
            None => 0,
        };

        // Load or create index
        let index = if mv2x_path.exists() {
            match Mv2xIndex::read_from(&mv2x_path) {
                Ok(idx) => idx,
                Err(_) => {
                    // Corrupt index — rebuild
                    tracing::warn!("corrupt .mv2x, rebuilding from .mv2d");
                    Self::build_index_from_mv2d(&mv2d_path)?
                }
            }
        } else {
            Self::build_index_from_mv2d(&mv2d_path)?
        };

        // Build inactive frames set
        let inactive_frames = Self::scan_inactive_frames(&mv2d_path)?;

        let mut store = Self {
            mv2d_path,
            mv2x_path,
            mv2d_file,
            index,
            next_frame_id,
            inactive_frames,
        };

        // Reconciliation check
        let mv2d_last = if next_frame_id > 0 { Some(next_frame_id - 1) } else { None };
        if store.index.needs_reconciliation(mv2d_last) {
            tracing::info!("index lag detected, running incremental catch-up");
            store.reconcile()?;
        }

        Ok(store)
    }

    /// Append a new frame. Returns the assigned frame_id.
    ///
    /// This only writes to `.mv2d` (JSONL append, µs-duration).
    /// Index update must be triggered separately via [`update_index`].
    pub fn put(&mut self, content: impl Into<String>) -> Result<u64> {
        let frame_id = self.next_frame_id;
        let frame = FrameJsonl::new(frame_id, content);
        let offset = frame_jsonl::write_frame_jsonl(&mut self.mv2d_file, &frame)?;

        self.index.add_frame(frame_id, offset, frame.created_at);
        self.next_frame_id += 1;
        Ok(frame_id)
    }

    /// Append a frame with full metadata.
    pub fn put_frame(&mut self, mut frame: FrameJsonl) -> Result<u64> {
        frame.frame_id = self.next_frame_id;
        let offset = frame_jsonl::write_frame_jsonl(&mut self.mv2d_file, &frame)?;

        // Track superseded frames as inactive
        if let Some(old_id) = frame.supersedes {
            self.inactive_frames.insert(old_id);
        }
        if frame.status == FrameStatus::Tombstoned {
            if let Some(old_id) = frame.supersedes {
                self.inactive_frames.insert(old_id);
            }
        }

        self.index.add_frame(frame.frame_id, offset, frame.created_at);
        self.next_frame_id += 1;
        Ok(frame.frame_id)
    }

    /// Delete a frame by appending a tombstone.
    pub fn delete(&mut self, target_frame_id: u64) -> Result<u64> {
        let tombstone = FrameJsonl::tombstone(self.next_frame_id, target_frame_id);
        self.put_frame(tombstone)
    }

    /// Update a frame by appending a new version with `supersedes`.
    pub fn update(&mut self, old_frame_id: u64, new_content: impl Into<String>) -> Result<u64> {
        let frame = FrameJsonl::new(self.next_frame_id, new_content)
            .with_supersedes(old_frame_id);
        self.put_frame(frame)
    }

    /// Read a single frame by ID from `.mv2d` using the offset index.
    pub fn get(&mut self, frame_id: u64) -> Result<Option<FrameJsonl>> {
        let offset = match self.index.offset_for(frame_id) {
            Some(o) => o,
            None => return Ok(None),
        };

        self.mv2d_file.seek(SeekFrom::Start(offset))?;
        let mut reader = BufReader::new(&self.mv2d_file);
        let mut line = String::new();
        reader.read_line(&mut line)?;

        if line.is_empty() {
            return Ok(None);
        }

        let frame: FrameJsonl = serde_json::from_str(line.trim())
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        Ok(Some(frame))
    }

    /// Get all active frames (excluding tombstoned and superseded).
    pub fn active_frames(&self) -> Result<Vec<FrameJsonl>> {
        let all = frame_jsonl::read_all_frames(&self.mv2d_path)?;
        let frames = all.into_iter()
            .map(|(_, f)| f)
            .filter(|f| f.is_active() && !self.inactive_frames.contains(&f.frame_id))
            .collect();
        Ok(frames)
    }

    /// Get the latest active frame (e.g., for prompts — "current version").
    pub fn latest_active(&self) -> Result<Option<FrameJsonl>> {
        let active = self.active_frames()?;
        Ok(active.into_iter().last())
    }

    /// Find frames by time range.
    pub fn frames_in_range(&self, start: i64, end: i64) -> Vec<u64> {
        self.index.frames_in_range(start, end)
            .into_iter()
            .filter(|id| !self.inactive_frames.contains(id))
            .collect()
    }

    // -- Search --

    /// Full-text search over frame contents.
    pub fn lex_search(&mut self, query: &str, limit: usize) -> Vec<crate::lex::LexSearchHit> {
        self.index.lex_search(query, limit)
    }

    /// Vector similarity search over frame embeddings.
    pub fn vec_search(&self, query: &[f32], limit: usize) -> Vec<crate::vec::VecSearchHit> {
        self.index.vec_search(query, limit)
    }

    /// Access the knowledge graph (LogicMesh).
    pub fn logic_mesh(&mut self) -> Option<&crate::types::logic_mesh::LogicMesh> {
        self.index.logic_mesh()
    }

    /// Access the knowledge graph mutably (for adding nodes/edges).
    pub fn logic_mesh_mut(&mut self) -> Option<&mut crate::types::logic_mesh::LogicMesh> {
        self.index.logic_mesh_mut()
    }

    /// Set or replace the knowledge graph.
    pub fn set_logic_mesh(&mut self, mesh: crate::types::logic_mesh::LogicMesh) {
        self.index.set_logic_mesh(mesh);
    }

    /// Add a single embedding vector for a frame. Call after computing the embedding.
    pub fn add_vec(&mut self, frame_id: u64, embedding: Vec<f32>) -> Result<()> {
        self.index.build_vec(std::iter::once((frame_id, embedding)))
    }

    /// Build the vector index from a batch of (frame_id, embedding) pairs.
    pub fn build_vec_index(
        &mut self,
        embeddings: impl Iterator<Item = (u64, Vec<f32>)>,
    ) -> Result<()> {
        self.index.build_vec(embeddings)
    }

    /// Rebuild the lex (full-text) index from all active frames in `.mv2d`.
    pub fn rebuild_lex_index(&mut self) -> Result<()> {
        let all = frame_jsonl::read_all_frames(&self.mv2d_path)?;
        let active: Vec<_> = all.into_iter()
            .map(|(_, f)| f)
            .filter(|f| f.is_active() && !self.inactive_frames.contains(&f.frame_id))
            .collect();
        self.index.build_lex(active.iter())
    }

    /// Set vec index from precomputed embeddings.
    pub fn set_vec_embeddings(
        &mut self,
        embeddings: impl Iterator<Item = (u64, Vec<f32>)>,
    ) -> Result<()> {
        self.index.build_vec(embeddings)
    }

    /// Persist the current index to `.mv2x`.
    pub fn flush_index(&mut self) -> Result<()> {
        self.index.write_to(&self.mv2x_path)
    }

    /// Full index rebuild from `.mv2d`.
    pub fn rebuild_index(&mut self) -> Result<()> {
        self.index = Self::build_index_from_mv2d(&self.mv2d_path)?;
        self.inactive_frames = Self::scan_inactive_frames(&self.mv2d_path)?;
        self.flush_index()
    }

    /// Incremental catch-up: index frames that are in `.mv2d` but not in `.mv2x`.
    pub fn reconcile(&mut self) -> Result<()> {
        let all_frames = frame_jsonl::read_all_frames(&self.mv2d_path)?;
        let last_indexed = self.index.header.last_indexed_frame_id.unwrap_or(0);

        for (offset, frame) in &all_frames {
            if frame.frame_id > last_indexed || self.index.offset_for(frame.frame_id).is_none() {
                self.index.add_frame(frame.frame_id, *offset, frame.created_at);
            }
            // Track inactive
            if let Some(old_id) = frame.supersedes {
                self.inactive_frames.insert(old_id);
            }
            if frame.status == FrameStatus::Tombstoned {
                if let Some(old_id) = frame.supersedes {
                    self.inactive_frames.insert(old_id);
                }
            }
        }

        self.flush_index()
    }

    /// Compact the `.mv2d` — remove tombstoned and superseded frames.
    ///
    /// Creates a new file, copies only active frames, rebuilds index.
    pub fn compact(&mut self) -> Result<CompactResult> {
        let active = self.active_frames()?;
        let original_count = self.next_frame_id;
        let active_count = active.len() as u64;

        // Write compacted file
        let tmp_path = self.mv2d_path.with_extension("mv2d.compact");
        {
            let mut tmp_file = frame_jsonl::open_mv2d(&tmp_path)?;
            for (new_id, frame) in active.into_iter().enumerate() {
                let mut compacted = frame;
                compacted.frame_id = new_id as u64;
                compacted.supersedes = None;
                frame_jsonl::write_frame_jsonl(&mut tmp_file, &compacted)?;
            }
        }

        // Atomic swap
        fs::rename(&tmp_path, &self.mv2d_path)?;

        // Reopen and rebuild
        self.mv2d_file = frame_jsonl::open_mv2d(&self.mv2d_path)?;
        self.next_frame_id = active_count;
        self.inactive_frames.clear();
        self.rebuild_index()?;

        Ok(CompactResult {
            original_frames: original_count,
            active_frames: active_count,
            removed_frames: original_count - active_count,
        })
    }

    /// Get the path to the `.mv2d` file.
    pub fn mv2d_path(&self) -> &Path {
        &self.mv2d_path
    }

    /// Get the path to the `.mv2x` file.
    pub fn mv2x_path(&self) -> &Path {
        &self.mv2x_path
    }

    /// Get the current frame count (including inactive).
    pub fn frame_count(&self) -> u64 {
        self.next_frame_id
    }

    /// Get the active frame count.
    pub fn active_frame_count(&self) -> u64 {
        self.next_frame_id - self.inactive_frames.len() as u64
    }

    /// Get the tombstone ratio (0.0 - 1.0).
    pub fn tombstone_ratio(&self) -> f64 {
        if self.next_frame_id == 0 {
            return 0.0;
        }
        self.inactive_frames.len() as f64 / self.next_frame_id as f64
    }

    // -- Private helpers --

    fn build_index_from_mv2d(mv2d_path: &Path) -> Result<Mv2xIndex> {
        let mut index = Mv2xIndex::new();
        if !mv2d_path.exists() {
            return Ok(index);
        }
        let frames = frame_jsonl::read_all_frames(mv2d_path)?;
        for (offset, frame) in &frames {
            index.add_frame(frame.frame_id, *offset, frame.created_at);
        }
        Ok(index)
    }

    fn scan_inactive_frames(mv2d_path: &Path) -> Result<HashSet<u64>> {
        let mut inactive = HashSet::new();
        if !mv2d_path.exists() {
            return Ok(inactive);
        }
        let frames = frame_jsonl::read_all_frames(mv2d_path)?;
        for (_, frame) in &frames {
            if let Some(old_id) = frame.supersedes {
                inactive.insert(old_id);
            }
            if frame.status == FrameStatus::Tombstoned {
                if let Some(old_id) = frame.supersedes {
                    inactive.insert(old_id);
                }
            }
        }
        Ok(inactive)
    }
}

/// Result of a compaction operation.
#[derive(Debug, Clone)]
pub struct CompactResult {
    pub original_frames: u64,
    pub active_frames: u64,
    pub removed_frames: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_dir() -> TempDir {
        TempDir::new().unwrap()
    }

    #[test]
    fn create_and_put() {
        let dir = test_dir();
        let path = dir.path().join("test.mv2d");

        let mut store = SplitMemvid::open(&path).unwrap();
        let id = store.put("Hello, world!").unwrap();
        assert_eq!(id, 0);

        let id2 = store.put("Second frame").unwrap();
        assert_eq!(id2, 1);

        assert_eq!(store.frame_count(), 2);
        assert_eq!(store.active_frame_count(), 2);
    }

    #[test]
    fn put_and_get() {
        let dir = test_dir();
        let path = dir.path().join("test.mv2d");

        let mut store = SplitMemvid::open(&path).unwrap();
        store.put("Frame zero").unwrap();
        store.put("Frame one").unwrap();
        store.flush_index().unwrap();

        let frame = store.get(0).unwrap().unwrap();
        assert_eq!(frame.content, "Frame zero");

        let frame = store.get(1).unwrap().unwrap();
        assert_eq!(frame.content, "Frame one");

        assert!(store.get(99).unwrap().is_none());
    }

    #[test]
    fn delete_creates_tombstone() {
        let dir = test_dir();
        let path = dir.path().join("test.mv2d");

        let mut store = SplitMemvid::open(&path).unwrap();
        store.put("To be deleted").unwrap();
        store.delete(0).unwrap();

        assert_eq!(store.frame_count(), 2); // original + tombstone
        assert_eq!(store.active_frame_count(), 1); // only tombstone is "active" by frame status
        assert!(store.inactive_frames.contains(&0));
    }

    #[test]
    fn update_supersedes() {
        let dir = test_dir();
        let path = dir.path().join("test.mv2d");

        let mut store = SplitMemvid::open(&path).unwrap();
        store.put("Version 1").unwrap();
        store.update(0, "Version 2").unwrap();

        assert!(store.inactive_frames.contains(&0));
        let active = store.active_frames().unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].content, "Version 2");
        assert_eq!(active[0].supersedes, Some(0));
    }

    #[test]
    fn reopen_with_reconciliation() {
        let dir = test_dir();
        let path = dir.path().join("test.mv2d");

        // Write some frames
        {
            let mut store = SplitMemvid::open(&path).unwrap();
            store.put("Frame A").unwrap();
            store.put("Frame B").unwrap();
            store.flush_index().unwrap();
        }

        // Reopen — should reconcile
        {
            let mut store = SplitMemvid::open(&path).unwrap();
            assert_eq!(store.frame_count(), 2);
            let frame = store.get(1).unwrap().unwrap();
            assert_eq!(frame.content, "Frame B");
        }
    }

    #[test]
    fn reopen_without_mv2x_rebuilds() {
        let dir = test_dir();
        let mv2d_path = dir.path().join("test.mv2d");
        let mv2x_path = dir.path().join("test.mv2x");

        // Write frames and flush index
        {
            let mut store = SplitMemvid::open(&mv2d_path).unwrap();
            store.put("Persistent").unwrap();
            store.flush_index().unwrap();
        }

        // Delete .mv2x
        fs::remove_file(&mv2x_path).unwrap();

        // Reopen — should rebuild index from .mv2d
        {
            let mut store = SplitMemvid::open(&mv2d_path).unwrap();
            assert_eq!(store.frame_count(), 1);
            let frame = store.get(0).unwrap().unwrap();
            assert_eq!(frame.content, "Persistent");
        }
    }

    #[test]
    fn compact_removes_inactive() {
        let dir = test_dir();
        let path = dir.path().join("test.mv2d");

        let mut store = SplitMemvid::open(&path).unwrap();
        store.put("Keep this").unwrap();
        store.put("Delete this").unwrap();
        store.put("Keep this too").unwrap();
        store.delete(1).unwrap();

        let result = store.compact().unwrap();
        assert_eq!(result.original_frames, 4); // 3 frames + 1 tombstone
        assert_eq!(result.active_frames, 2);
        assert_eq!(result.removed_frames, 2);

        // Verify compacted state
        let active = store.active_frames().unwrap();
        assert_eq!(active.len(), 2);
        assert_eq!(active[0].content, "Keep this");
        assert_eq!(active[1].content, "Keep this too");
    }

    #[test]
    fn tombstone_ratio() {
        let dir = test_dir();
        let path = dir.path().join("test.mv2d");

        let mut store = SplitMemvid::open(&path).unwrap();
        assert_eq!(store.tombstone_ratio(), 0.0);

        store.put("A").unwrap();
        store.put("B").unwrap();
        store.put("C").unwrap();
        store.delete(0).unwrap();
        store.delete(1).unwrap();

        // 5 frames total, 2 inactive (0 and 1)
        assert!((store.tombstone_ratio() - 0.4).abs() < 0.01);
    }

    #[test]
    fn latest_active_returns_last() {
        let dir = test_dir();
        let path = dir.path().join("test.mv2d");

        let mut store = SplitMemvid::open(&path).unwrap();
        store.put("First").unwrap();
        store.put("Second").unwrap();
        store.put("Third").unwrap();

        let latest = store.latest_active().unwrap().unwrap();
        assert_eq!(latest.content, "Third");
    }

    #[test]
    fn put_frame_with_tags() {
        let dir = test_dir();
        let path = dir.path().join("test.mv2d");

        let mut store = SplitMemvid::open(&path).unwrap();
        let frame = FrameJsonl::new(0, "Tagged content")
            .with_uri("mv2://memory/episodic/001")
            .with_title("Test")
            .with_tag("agent", "anubis")
            .with_tag("sector", "episodic");
        store.put_frame(frame).unwrap();

        let loaded = store.get(0).unwrap().unwrap();
        assert_eq!(loaded.tags.get("agent").unwrap(), "anubis");
        assert_eq!(loaded.uri.as_deref(), Some("mv2://memory/episodic/001"));
    }
}
