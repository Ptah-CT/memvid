//! NAPI bindings for `memvid-core` split-file backend.
//!
//! Exposes `SplitMemvid` to Node.js as a native addon.

use napi::bindgen_prelude::*;
use napi_derive::napi;

use memvid_core::split::{FrameJsonl, FrameStatus, SplitMemvid};

/// A single frame in the split-file store.
#[napi(object)]
pub struct JsFrame {
    pub frame_id: i64,
    pub uri: Option<String>,
    pub title: Option<String>,
    pub created_at: i64,
    pub status: String,
    pub supersedes: Option<i64>,
    pub tags: serde_json::Value,
    pub content: String,
}

impl From<FrameJsonl> for JsFrame {
    fn from(f: FrameJsonl) -> Self {
        Self {
            frame_id: f.frame_id as i64,
            uri: f.uri,
            title: f.title,
            created_at: f.created_at,
            status: match f.status {
                FrameStatus::Active => "active".to_string(),
                FrameStatus::Tombstoned => "tombstoned".to_string(),
            },
            supersedes: f.supersedes.map(|id| id as i64),
            tags: serde_json::to_value(&f.tags).unwrap_or_default(),
            content: f.content,
        }
    }
}

/// Search hit from full-text search.
#[napi(object)]
pub struct JsLexHit {
    pub frame_id: i64,
    pub score: f64,
    pub match_count: i64,
}

/// Search hit from vector similarity search.
#[napi(object)]
pub struct JsVecHit {
    pub frame_id: i64,
    pub score: f64,
}

/// Result of a compaction operation.
#[napi(object)]
pub struct JsCompactResult {
    pub original_frames: i64,
    pub active_frames: i64,
    pub removed_frames: i64,
}

/// Split-file Memvid store: `.mv2d` (JSONL) + `.mv2x` (binary index).
#[napi]
pub struct SplitStore {
    inner: SplitMemvid,
}

#[napi]
impl SplitStore {
    /// Open or create a split store at the given `.mv2d` path.
    /// The `.mv2x` index file is created alongside automatically.
    #[napi(constructor)]
    pub fn new(mv2d_path: String) -> Result<Self> {
        let inner = SplitMemvid::open(&mv2d_path)
            .map_err(|e| Error::from_reason(format!("{e}")))?;
        Ok(Self { inner })
    }

    /// Append a text frame. Returns the assigned frame_id.
    #[napi]
    pub fn put(&mut self, content: String) -> Result<i64> {
        let id = self.inner.put(content)
            .map_err(|e| Error::from_reason(format!("{e}")))?;
        Ok(id as i64)
    }

    /// Append a frame with full metadata.
    #[napi]
    pub fn put_frame(
        &mut self,
        content: String,
        uri: Option<String>,
        title: Option<String>,
        tags: Option<serde_json::Value>,
    ) -> Result<i64> {
        let mut frame = FrameJsonl::new(0, content);
        if let Some(u) = uri {
            frame = frame.with_uri(u);
        }
        if let Some(t) = title {
            frame = frame.with_title(t);
        }
        if let Some(serde_json::Value::Object(map)) = tags {
            for (k, v) in map {
                if let serde_json::Value::String(vs) = v {
                    frame = frame.with_tag(k, vs);
                }
            }
        }
        let id = self.inner.put_frame(frame)
            .map_err(|e| Error::from_reason(format!("{e}")))?;
        Ok(id as i64)
    }

    /// Get a frame by ID.
    #[napi]
    pub fn get(&mut self, frame_id: i64) -> Result<Option<JsFrame>> {
        let frame = self.inner.get(frame_id as u64)
            .map_err(|e| Error::from_reason(format!("{e}")))?;
        Ok(frame.map(JsFrame::from))
    }

    /// Delete a frame (append tombstone).
    #[napi]
    pub fn delete(&mut self, frame_id: i64) -> Result<i64> {
        let id = self.inner.delete(frame_id as u64)
            .map_err(|e| Error::from_reason(format!("{e}")))?;
        Ok(id as i64)
    }

    /// Update a frame (append new version with supersedes).
    #[napi]
    pub fn update(&mut self, old_frame_id: i64, new_content: String) -> Result<i64> {
        let id = self.inner.update(old_frame_id as u64, new_content)
            .map_err(|e| Error::from_reason(format!("{e}")))?;
        Ok(id as i64)
    }

    /// Get all active frames.
    #[napi]
    pub fn active_frames(&self) -> Result<Vec<JsFrame>> {
        let frames = self.inner.active_frames()
            .map_err(|e| Error::from_reason(format!("{e}")))?;
        Ok(frames.into_iter().map(JsFrame::from).collect())
    }

    /// Get the latest active frame (e.g., current prompt version).
    #[napi]
    pub fn latest_active(&self) -> Result<Option<JsFrame>> {
        let frame = self.inner.latest_active()
            .map_err(|e| Error::from_reason(format!("{e}")))?;
        Ok(frame.map(JsFrame::from))
    }

    /// Full-text search.
    #[napi]
    pub fn lex_search(&mut self, query: String, limit: i64) -> Vec<JsLexHit> {
        self.inner.lex_search(&query, limit as usize)
            .into_iter()
            .map(|h| JsLexHit {
                frame_id: h.frame_id as i64,
                score: f64::from(h.score),
                match_count: h.match_count as i64,
            })
            .collect()
    }

    /// Vector similarity search.
    #[napi]
    pub fn vec_search(&self, query: Vec<f64>, limit: i64) -> Vec<JsVecHit> {
        let query_f32: Vec<f32> = query.iter().map(|&v| v as f32).collect();
        self.inner.vec_search(&query_f32, limit as usize)
            .into_iter()
            .map(|h| JsVecHit {
                frame_id: h.frame_id as i64,
                score: f64::from(h.distance),
            })
            .collect()
    }

    /// Add a precomputed embedding vector for a single frame.
    #[napi]
    pub fn add_vec(&mut self, frame_id: i64, embedding: Vec<f64>) -> Result<()> {
        let embedding_f32: Vec<f32> = embedding.iter().map(|&v| v as f32).collect();
        self.inner.add_vec(frame_id as u64, embedding_f32)
            .map_err(|e| Error::from_reason(format!("{e}")))
    }

    /// Build the vector index from a batch of (frame_id, embedding) pairs.
    #[napi]
    pub fn build_vec_index(&mut self, frame_ids: Vec<i64>, embeddings: Vec<Vec<f64>>) -> Result<()> {
        let pairs = frame_ids.into_iter().zip(embeddings).map(|(fid, emb)| {
            (fid as u64, emb.iter().map(|&v| v as f32).collect::<Vec<f32>>())
        });
        self.inner.build_vec_index(pairs)
            .map_err(|e| Error::from_reason(format!("{e}")))
    }

    /// Rebuild the full-text (lex) index from all active frames.
    #[napi]
    pub fn rebuild_lex_index(&mut self) -> Result<()> {
        self.inner.rebuild_lex_index()
            .map_err(|e| Error::from_reason(format!("{e}")))
    }

    /// Persist the current index to `.mv2x`.
    #[napi]
    pub fn flush_index(&mut self) -> Result<()> {
        self.inner.flush_index()
            .map_err(|e| Error::from_reason(format!("{e}")))
    }

    /// Full index rebuild from `.mv2d`.
    #[napi]
    pub fn rebuild_index(&mut self) -> Result<()> {
        self.inner.rebuild_index()
            .map_err(|e| Error::from_reason(format!("{e}")))
    }

    /// Compact: remove tombstoned and superseded frames.
    #[napi]
    pub fn compact(&mut self) -> Result<JsCompactResult> {
        let r = self.inner.compact()
            .map_err(|e| Error::from_reason(format!("{e}")))?;
        Ok(JsCompactResult {
            original_frames: r.original_frames as i64,
            active_frames: r.active_frames as i64,
            removed_frames: r.removed_frames as i64,
        })
    }

    /// Total frame count (including inactive).
    #[napi(getter)]
    pub fn frame_count(&self) -> i64 {
        self.inner.frame_count() as i64
    }

    /// Active frame count.
    #[napi(getter)]
    pub fn active_frame_count(&self) -> i64 {
        self.inner.active_frame_count() as i64
    }

    /// Tombstone ratio (0.0 - 1.0).
    #[napi(getter)]
    pub fn tombstone_ratio(&self) -> f64 {
        self.inner.tombstone_ratio()
    }
}
