//! `.mv2x` index file — binary indices derived from `.mv2d` data.
//!
//! Contains: frame offset index, time index, and metadata header.
//! Lex (Tantivy) and Vec (HNSW) indices are optional features.
//!
//! This file is a derived artifact — deletable and rebuildable from `.mv2d`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::Result;

/// Magic bytes for `.mv2x` files.
const MV2X_MAGIC: &[u8; 4] = b"MV2X";

/// Current spec version.
const MV2X_VERSION: u16 = 1;

/// Header stored at the beginning of `.mv2x`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mv2xHeader {
    /// Embedding model name (for mismatch detection).
    pub embedding_model: Option<String>,
    /// Embedding dimensions.
    pub embedding_dim: Option<u32>,
    /// Number of frames indexed.
    pub frame_count: u64,
    /// Last frame_id that was indexed (for reconciliation).
    pub last_indexed_frame_id: Option<u64>,
    /// Generation counter (incremented on each rebuild/update).
    pub generation: u64,
}

impl Default for Mv2xHeader {
    fn default() -> Self {
        Self {
            embedding_model: None,
            embedding_dim: None,
            frame_count: 0,
            last_indexed_frame_id: None,
            generation: 0,
        }
    }
}

/// Frame offset entry: maps frame_id to byte offset in `.mv2d`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameOffset {
    pub frame_id: u64,
    pub byte_offset: u64,
}

/// Time index entry for chronological queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeEntry {
    pub frame_id: u64,
    pub timestamp: i64,
}

/// Complete `.mv2x` index state (in-memory representation).
#[derive(Debug, Clone, Default)]
pub struct Mv2xIndex {
    pub header: Mv2xHeader,
    /// Frame ID → byte offset in `.mv2d`.
    pub offsets: Vec<FrameOffset>,
    /// Chronologically sorted time entries.
    pub time_entries: Vec<TimeEntry>,
}

impl Mv2xIndex {
    /// Create an empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a frame to the index.
    pub fn add_frame(&mut self, frame_id: u64, byte_offset: u64, timestamp: i64) {
        self.offsets.push(FrameOffset { frame_id, byte_offset });
        self.time_entries.push(TimeEntry { frame_id, timestamp });
        self.header.frame_count = self.offsets.len() as u64;
        self.header.last_indexed_frame_id = Some(frame_id);
    }

    /// Look up the byte offset for a frame_id.
    pub fn offset_for(&self, frame_id: u64) -> Option<u64> {
        self.offsets.iter()
            .find(|o| o.frame_id == frame_id)
            .map(|o| o.byte_offset)
    }

    /// Find frame_ids within a time range.
    pub fn frames_in_range(&self, start: i64, end: i64) -> Vec<u64> {
        self.time_entries.iter()
            .filter(|e| e.timestamp >= start && e.timestamp <= end)
            .map(|e| e.frame_id)
            .collect()
    }

    /// Serialize the index to a `.mv2x` file.
    pub fn write_to(&self, path: &Path) -> Result<()> {
        let file = File::create(path)?;
        let mut w = BufWriter::new(file);

        // Magic + version
        w.write_all(MV2X_MAGIC)?;
        w.write_all(&MV2X_VERSION.to_le_bytes())?;

        // Header (bincode)
        let header_bytes = bincode::serde::encode_to_vec(&self.header, bincode_config())
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        w.write_all(&(header_bytes.len() as u32).to_le_bytes())?;
        w.write_all(&header_bytes)?;

        // Offsets (bincode)
        let offsets_bytes = bincode::serde::encode_to_vec(&self.offsets, bincode_config())
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        w.write_all(&(offsets_bytes.len() as u32).to_le_bytes())?;
        w.write_all(&offsets_bytes)?;

        // Time entries (bincode)
        let time_bytes = bincode::serde::encode_to_vec(&self.time_entries, bincode_config())
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        w.write_all(&(time_bytes.len() as u32).to_le_bytes())?;
        w.write_all(&time_bytes)?;

        w.flush()?;
        Ok(())
    }

    /// Deserialize an index from a `.mv2x` file.
    pub fn read_from(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let mut r = BufReader::new(file);

        // Magic
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != MV2X_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid .mv2x magic bytes",
            ).into());
        }

        // Version
        let mut version_bytes = [0u8; 2];
        r.read_exact(&mut version_bytes)?;
        let version = u16::from_le_bytes(version_bytes);
        if version > MV2X_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported .mv2x version: {version}"),
            ).into());
        }

        // Header
        let header = read_section::<Mv2xHeader>(&mut r)?;

        // Offsets
        let offsets = read_section::<Vec<FrameOffset>>(&mut r)?;

        // Time entries
        let time_entries = read_section::<Vec<TimeEntry>>(&mut r)?;

        Ok(Self { header, offsets, time_entries })
    }

    /// Check if reconciliation is needed against a `.mv2d` file's last frame_id.
    pub fn needs_reconciliation(&self, mv2d_last_frame_id: Option<u64>) -> bool {
        match (self.header.last_indexed_frame_id, mv2d_last_frame_id) {
            (None, None) => false,           // Both empty
            (None, Some(_)) => true,         // Index empty, data has frames
            (Some(_), None) => true,         // Index has frames, data empty (corrupt?)
            (Some(idx), Some(data)) => idx != data,  // Diverged
        }
    }
}

fn read_section<T: for<'a> serde::Deserialize<'a>>(reader: &mut BufReader<File>) -> Result<T> {
    let mut len_bytes = [0u8; 4];
    reader.read_exact(&mut len_bytes)?;
    let len = u32::from_le_bytes(len_bytes) as usize;

    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;

    let (value, _) = bincode::serde::decode_from_slice::<T, _>(&buf, bincode_config())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    Ok(value)
}

fn bincode_config() -> impl bincode::config::Config {
    bincode::config::standard()
        .with_fixed_int_encoding()
        .with_little_endian()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn roundtrip_empty_index() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.mv2x");

        let index = Mv2xIndex::new();
        index.write_to(&path).unwrap();

        let loaded = Mv2xIndex::read_from(&path).unwrap();
        assert_eq!(loaded.header.frame_count, 0);
        assert!(loaded.offsets.is_empty());
        assert!(loaded.time_entries.is_empty());
    }

    #[test]
    fn roundtrip_with_frames() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.mv2x");

        let mut index = Mv2xIndex::new();
        index.header.embedding_model = Some("qwen3-embedding-8b".to_string());
        index.header.embedding_dim = Some(4096);
        index.add_frame(0, 0, 1744544400);
        index.add_frame(1, 256, 1744545000);
        index.add_frame(2, 512, 1744545600);

        index.write_to(&path).unwrap();

        let loaded = Mv2xIndex::read_from(&path).unwrap();
        assert_eq!(loaded.header.frame_count, 3);
        assert_eq!(loaded.header.embedding_model.as_deref(), Some("qwen3-embedding-8b"));
        assert_eq!(loaded.header.embedding_dim, Some(4096));
        assert_eq!(loaded.header.last_indexed_frame_id, Some(2));
        assert_eq!(loaded.offset_for(1), Some(256));
    }

    #[test]
    fn time_range_query() {
        let mut index = Mv2xIndex::new();
        index.add_frame(0, 0, 100);
        index.add_frame(1, 50, 200);
        index.add_frame(2, 100, 300);
        index.add_frame(3, 150, 400);

        let in_range = index.frames_in_range(150, 350);
        assert_eq!(in_range, vec![1, 2]);
    }

    #[test]
    fn reconciliation_check() {
        let mut index = Mv2xIndex::new();
        assert!(!index.needs_reconciliation(None)); // Both empty

        index.add_frame(0, 0, 100);
        assert!(index.needs_reconciliation(None)); // Index has data, mv2d empty
        assert!(!index.needs_reconciliation(Some(0))); // In sync
        assert!(index.needs_reconciliation(Some(5))); // mv2d ahead
    }
}
