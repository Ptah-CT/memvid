//! `.mv2x` index file — binary indices derived from `.mv2d` data.
//!
//! Contains: frame offset index, time index, lex index, vec index, and LogicMesh.
//! This file is a derived artifact — deletable and rebuildable from `.mv2d`.

use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::lex::{LexIndex, LexIndexArtifact, LexIndexBuilder, LexSearchHit};
use crate::types::logic_mesh::LogicMesh;
use crate::vec::{VecIndexBuilder, VecSearchHit};
use crate::Result;

/// Magic bytes for `.mv2x` files.
const MV2X_MAGIC: &[u8; 4] = b"MV2X";

/// Current spec version.
const MV2X_VERSION: u16 = 2;

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
#[derive(Debug, Default)]
pub struct Mv2xIndex {
    pub header: Mv2xHeader,
    /// Frame ID → byte offset in `.mv2d`.
    pub offsets: Vec<FrameOffset>,
    /// Chronologically sorted time entries.
    pub time_entries: Vec<TimeEntry>,
    /// Lex (full-text) index — serialized blob.
    lex_bytes: Option<Vec<u8>>,
    /// Lex index (decoded, in-memory).
    lex_index: Option<LexIndex>,
    /// Vec (HNSW) index — serialized blob.
    vec_bytes: Option<Vec<u8>>,
    /// LogicMesh (knowledge graph) — serialized blob.
    mesh_bytes: Option<Vec<u8>>,
    /// LogicMesh (decoded, in-memory).
    logic_mesh: Option<LogicMesh>,
}

impl Mv2xIndex {
    /// Create an empty index.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a frame to the offset + time index.
    pub fn add_frame(&mut self, frame_id: u64, byte_offset: u64, timestamp: i64) {
        self.offsets.push(FrameOffset {
            frame_id,
            byte_offset,
        });
        self.time_entries.push(TimeEntry {
            frame_id,
            timestamp,
        });
        self.header.frame_count = self.offsets.len() as u64;
        self.header.last_indexed_frame_id = Some(frame_id);
    }

    /// Look up the byte offset for a frame_id.
    pub fn offset_for(&self, frame_id: u64) -> Option<u64> {
        self.offsets
            .iter()
            .find(|o| o.frame_id == frame_id)
            .map(|o| o.byte_offset)
    }

    /// Find frame_ids within a time range.
    pub fn frames_in_range(&self, start: i64, end: i64) -> Vec<u64> {
        self.time_entries
            .iter()
            .filter(|e| e.timestamp >= start && e.timestamp <= end)
            .map(|e| e.frame_id)
            .collect()
    }

    // -- Lex (full-text search) --

    /// Set the lex index from a built artifact.
    pub fn set_lex(&mut self, artifact: LexIndexArtifact) {
        self.lex_bytes = Some(artifact.bytes);
        self.lex_index = None; // Will be decoded lazily
    }

    /// Full-text search over indexed frames.
    pub fn lex_search(&mut self, query: &str, limit: usize) -> Vec<LexSearchHit> {
        if self.lex_index.is_none() {
            if let Some(bytes) = &self.lex_bytes {
                self.lex_index = LexIndex::decode(bytes).ok();
            }
        }
        match &self.lex_index {
            Some(idx) => idx.search(query, limit),
            None => Vec::new(),
        }
    }

    /// Build and set the lex index from frames.
    pub fn build_lex<'a>(
        &mut self,
        frames: impl Iterator<Item = &'a super::frame_jsonl::FrameJsonl>,
    ) -> Result<()> {
        let mut builder = LexIndexBuilder::new();
        for frame in frames {
            if !frame.is_active() {
                continue;
            }
            builder.add_document(
                frame.frame_id,
                frame.uri.as_deref().unwrap_or(""),
                frame.title.as_deref(),
                &frame.content,
                &frame.tags,
            );
        }
        let artifact = builder.finish()?;
        self.set_lex(artifact);
        Ok(())
    }

    // -- Vec (vector/embedding search) --

    /// Set the vec index from a built artifact bytes.
    pub fn set_vec_bytes(&mut self, bytes: Vec<u8>) {
        self.vec_bytes = Some(bytes);
    }

    /// Build and set vec index from frame embeddings.
    pub fn build_vec(
        &mut self,
        embeddings: impl Iterator<Item = (u64, Vec<f32>)>,
    ) -> Result<()> {
        let mut builder = VecIndexBuilder::new();
        for (frame_id, embedding) in embeddings {
            builder.add_document(frame_id, embedding);
        }
        let artifact = builder.finish()?;
        self.vec_bytes = Some(artifact.bytes);
        Ok(())
    }

    /// Vector similarity search. Returns (frame_id, score) pairs.
    pub fn vec_search(&self, query: &[f32], limit: usize) -> Vec<VecSearchHit> {
        let bytes = match &self.vec_bytes {
            Some(b) => b,
            None => return Vec::new(),
        };
        let index = match crate::vec::VecIndex::decode(bytes) {
            Ok(idx) => idx,
            Err(_) => return Vec::new(),
        };
        index.search(query, limit)
    }

    // -- LogicMesh (knowledge graph) --

    /// Set the logic mesh.
    pub fn set_logic_mesh(&mut self, mesh: LogicMesh) {
        self.mesh_bytes = None; // Will be serialized on write
        self.logic_mesh = Some(mesh);
    }

    /// Get the logic mesh (decoded).
    pub fn logic_mesh(&mut self) -> Option<&LogicMesh> {
        if self.logic_mesh.is_none() {
            if let Some(bytes) = &self.mesh_bytes {
                self.logic_mesh = LogicMesh::deserialize(bytes).ok();
            }
        }
        self.logic_mesh.as_ref()
    }

    /// Get a mutable reference to the logic mesh.
    pub fn logic_mesh_mut(&mut self) -> Option<&mut LogicMesh> {
        // Ensure decoded
        if self.logic_mesh.is_none() {
            if let Some(bytes) = &self.mesh_bytes {
                self.logic_mesh = LogicMesh::deserialize(bytes).ok();
            }
        }
        self.logic_mesh.as_mut()
    }

    // -- Reconciliation --

    /// Check if reconciliation is needed against a `.mv2d` file's last frame_id.
    pub fn needs_reconciliation(&self, mv2d_last_frame_id: Option<u64>) -> bool {
        match (self.header.last_indexed_frame_id, mv2d_last_frame_id) {
            (None, None) => false,
            (None, Some(_)) => true,
            (Some(_), None) => true,
            (Some(idx), Some(data)) => idx != data,
        }
    }

    // -- Serialization --

    /// Serialize the complete index to a `.mv2x` file.
    pub fn write_to(&mut self, path: &Path) -> Result<()> {
        // Serialize logic mesh if present but not yet serialized
        if self.mesh_bytes.is_none() {
            if let Some(mesh) = &self.logic_mesh {
                if !mesh.is_empty() {
                    self.mesh_bytes = Some(mesh.serialize()?);
                }
            }
        }

        let file = File::create(path)?;
        let mut w = BufWriter::new(file);

        // Magic + version
        w.write_all(MV2X_MAGIC)?;
        w.write_all(&MV2X_VERSION.to_le_bytes())?;

        // Header
        write_section(&mut w, &self.header)?;

        // Offsets
        write_section(&mut w, &self.offsets)?;

        // Time entries
        write_section(&mut w, &self.time_entries)?;

        // Lex index (raw bytes, length-prefixed)
        write_blob(&mut w, self.lex_bytes.as_deref())?;

        // Vec index (raw bytes, length-prefixed)
        write_blob(&mut w, self.vec_bytes.as_deref())?;

        // LogicMesh (raw bytes, length-prefixed)
        write_blob(&mut w, self.mesh_bytes.as_deref())?;

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
            return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid .mv2x magic bytes")
                .into());
        }

        // Version
        let mut version_bytes = [0u8; 2];
        r.read_exact(&mut version_bytes)?;
        let version = u16::from_le_bytes(version_bytes);
        if version > MV2X_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported .mv2x version: {version}"),
            )
            .into());
        }

        // Header
        let header = read_section::<Mv2xHeader>(&mut r)?;

        // Offsets
        let offsets = read_section::<Vec<FrameOffset>>(&mut r)?;

        // Time entries
        let time_entries = read_section::<Vec<TimeEntry>>(&mut r)?;

        // Lex index blob
        let lex_bytes = read_blob(&mut r)?;

        // Vec index blob
        let vec_bytes = read_blob(&mut r)?;

        // LogicMesh blob
        let mesh_bytes = read_blob(&mut r)?;

        Ok(Self {
            header,
            offsets,
            time_entries,
            lex_bytes,
            lex_index: None,
            vec_bytes,
            mesh_bytes,
            logic_mesh: None,
        })
    }
}

// -- Wire helpers --

fn write_section<T: serde::Serialize>(w: &mut BufWriter<File>, data: &T) -> Result<()> {
    let bytes = bincode::serde::encode_to_vec(data, bincode_config())
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    w.write_all(&(bytes.len() as u32).to_le_bytes())?;
    w.write_all(&bytes)?;
    Ok(())
}

fn write_blob(w: &mut BufWriter<File>, data: Option<&[u8]>) -> Result<()> {
    match data {
        Some(bytes) => {
            w.write_all(&(bytes.len() as u32).to_le_bytes())?;
            w.write_all(bytes)?;
        }
        None => {
            w.write_all(&0u32.to_le_bytes())?;
        }
    }
    Ok(())
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

fn read_blob(reader: &mut BufReader<File>) -> Result<Option<Vec<u8>>> {
    let mut len_bytes = [0u8; 4];
    reader.read_exact(&mut len_bytes)?;
    let len = u32::from_le_bytes(len_bytes) as usize;
    if len == 0 {
        return Ok(None);
    }
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;
    Ok(Some(buf))
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

        let mut index = Mv2xIndex::new();
        index.write_to(&path).unwrap();

        let loaded = Mv2xIndex::read_from(&path).unwrap();
        assert_eq!(loaded.header.frame_count, 0);
        assert!(loaded.offsets.is_empty());
        assert!(loaded.time_entries.is_empty());
        assert!(loaded.lex_bytes.is_none());
        assert!(loaded.vec_bytes.is_none());
        assert!(loaded.mesh_bytes.is_none());
    }

    #[test]
    fn roundtrip_with_frames_and_header() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.mv2x");

        let mut index = Mv2xIndex::new();
        index.header.embedding_model = Some("qwen3-embedding-8b".to_string());
        index.header.embedding_dim = Some(4096);
        index.add_frame(0, 0, 1_744_544_400);
        index.add_frame(1, 256, 1_744_545_000);
        index.add_frame(2, 512, 1_744_545_600);

        index.write_to(&path).unwrap();

        let loaded = Mv2xIndex::read_from(&path).unwrap();
        assert_eq!(loaded.header.frame_count, 3);
        assert_eq!(
            loaded.header.embedding_model.as_deref(),
            Some("qwen3-embedding-8b")
        );
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
        assert!(!index.needs_reconciliation(None));

        index.add_frame(0, 0, 100);
        assert!(index.needs_reconciliation(None));
        assert!(!index.needs_reconciliation(Some(0)));
        assert!(index.needs_reconciliation(Some(5)));
    }

    #[test]
    fn lex_build_and_search() {
        use super::super::frame_jsonl::FrameJsonl;

        let frames = vec![
            FrameJsonl::new(0, "SurrealDB drops connections after 32 minutes idle"),
            FrameJsonl::new(1, "PQC ML-DSA-87 keys rotated successfully"),
            FrameJsonl::new(2, "WebSocket connection timeout debugging notes"),
        ];

        let mut index = Mv2xIndex::new();
        for (i, f) in frames.iter().enumerate() {
            index.add_frame(f.frame_id, i as u64 * 100, f.created_at);
        }
        index.build_lex(frames.iter()).unwrap();

        let hits = index.lex_search("connection timeout", 10);
        assert!(!hits.is_empty());
        // Frame 2 should score highest (has both "connection" and "timeout")
        assert_eq!(hits[0].frame_id, 2);
    }

    #[test]
    fn lex_roundtrip_through_file() {
        use super::super::frame_jsonl::FrameJsonl;

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("lex.mv2x");

        let frames = vec![
            FrameJsonl::new(0, "Filesystem based persistence"),
            FrameJsonl::new(1, "CORAL directory structure with symlinks"),
        ];

        let mut index = Mv2xIndex::new();
        for (i, f) in frames.iter().enumerate() {
            index.add_frame(f.frame_id, i as u64 * 100, f.created_at);
        }
        index.build_lex(frames.iter()).unwrap();
        index.write_to(&path).unwrap();

        let mut loaded = Mv2xIndex::read_from(&path).unwrap();
        let hits = loaded.lex_search("symlinks", 5);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].frame_id, 1);
    }

    #[test]
    fn logic_mesh_roundtrip() {
        use crate::types::logic_mesh::{
            compute_node_id, EntityKind, LinkType, LogicMesh, MeshEdge, MeshNode,
        };

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("mesh.mv2x");

        let mut mesh = LogicMesh::new();
        mesh.merge_node(MeshNode::new(
            "anubis".to_string(),
            "Anubis".to_string(),
            EntityKind::Other,
            0.95,
            0,
            0,
            6,
        ));
        mesh.merge_node(MeshNode::new(
            "surrealdb".to_string(),
            "SurrealDB".to_string(),
            EntityKind::Product,
            0.88,
            0,
            10,
            9,
        ));
        let anubis_id = compute_node_id("anubis", EntityKind::Other);
        let surreal_id = compute_node_id("surrealdb", EntityKind::Product);
        mesh.merge_edge(MeshEdge::new(
            anubis_id,
            surreal_id,
            LinkType::Related,
            0.9,
            0,
        ));
        mesh.finalize();

        let mut index = Mv2xIndex::new();
        index.set_logic_mesh(mesh);
        index.write_to(&path).unwrap();

        let mut loaded = Mv2xIndex::read_from(&path).unwrap();
        let loaded_mesh = loaded.logic_mesh().unwrap();
        assert_eq!(loaded_mesh.stats().node_count, 2);
        assert_eq!(loaded_mesh.stats().edge_count, 1);

        let results = loaded_mesh.follow("Anubis", "related", 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].node, "SurrealDB");
    }
}
