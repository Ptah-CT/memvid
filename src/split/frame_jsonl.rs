//! JSONL frame serialization for `.mv2d` files.
//!
//! Each line in a `.mv2d` file is a self-contained JSON object representing one frame.
//! Fields are compatible with the MV2 Spec v2.1 Frame structure.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::Result;

/// A single frame in JSONL format — one line in a `.mv2d` file.
///
/// This is the plaintext representation of a Memvid frame. Content is stored
/// uncompressed for human readability and grep-ability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameJsonl {
    /// Monotonically increasing ID, unique per file.
    pub frame_id: u64,
    /// Hierarchical URI (`mv2://path/to/content`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
    /// Human-readable title.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Unix timestamp in seconds.
    pub created_at: i64,
    /// Frame status: "active" or "tombstoned".
    pub status: FrameStatus,
    /// If this frame supersedes an older frame (for updates).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub supersedes: Option<u64>,
    /// Free-form key-value tags.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub tags: HashMap<String, String>,
    /// Plaintext content (uncompressed, human-readable).
    pub content: String,
}

/// Frame lifecycle status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FrameStatus {
    Active,
    Tombstoned,
}

impl FrameJsonl {
    /// Create a new active frame with the next ID.
    pub fn new(frame_id: u64, content: impl Into<String>) -> Self {
        Self {
            frame_id,
            uri: None,
            title: None,
            created_at: now_unix(),
            status: FrameStatus::Active,
            supersedes: None,
            tags: HashMap::new(),
            content: content.into(),
        }
    }

    /// Set the URI.
    pub fn with_uri(mut self, uri: impl Into<String>) -> Self {
        self.uri = Some(uri.into());
        self
    }

    /// Set the title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Add a tag.
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }

    /// Mark this frame as superseding another.
    pub fn with_supersedes(mut self, old_frame_id: u64) -> Self {
        self.supersedes = Some(old_frame_id);
        self
    }

    /// Create a tombstone frame (marks a frame as deleted).
    pub fn tombstone(frame_id: u64, target_frame_id: u64) -> Self {
        Self {
            frame_id,
            uri: None,
            title: None,
            created_at: now_unix(),
            status: FrameStatus::Tombstoned,
            supersedes: Some(target_frame_id),
            tags: HashMap::new(),
            content: String::new(),
        }
    }

    /// Check if this frame is active (not tombstoned or superseded).
    pub fn is_active(&self) -> bool {
        self.status == FrameStatus::Active
    }
}

/// Append a single frame to a `.mv2d` file. Caller must hold flock().
///
/// Returns the byte offset where the frame was written (for offset index).
pub fn write_frame_jsonl(file: &mut File, frame: &FrameJsonl) -> Result<u64> {
    let offset = file.seek(SeekFrom::End(0))?;
    let mut writer = BufWriter::new(&*file);
    serde_json::to_writer(&mut writer, frame)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(offset)
}

/// Read all frames from a `.mv2d` file.
///
/// Returns frames with their byte offsets for building the offset index.
pub fn read_all_frames(path: &Path) -> Result<Vec<(u64, FrameJsonl)>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut frames = Vec::new();
    let mut offset: u64 = 0;

    for line_result in reader.lines() {
        let line = line_result?;
        if line.is_empty() {
            offset += 1; // newline only
            continue;
        }
        let frame: FrameJsonl = serde_json::from_str(&line)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        frames.push((offset, frame));
        offset += line.len() as u64 + 1; // +1 for newline
    }

    Ok(frames)
}

/// Read the last frame_id from a `.mv2d` file (for next-ID determination).
///
/// Scans backwards in fixed-size chunks until two newlines are found (or the
/// start of the file is reached). This tolerates arbitrarily long lines —
/// earlier implementations capped the read at 4 KB / 64 KB and silently
/// truncated lines longer than that, which then failed JSON parsing.
pub fn last_frame_id(path: &Path) -> Result<Option<u64>> {
    let file = File::open(path)?;
    let metadata = file.metadata()?;
    let file_size = metadata.len();

    if file_size == 0 {
        return Ok(None);
    }

    const CHUNK: u64 = 8192;
    let mut reader = BufReader::new(file);
    let mut tail: Vec<u8> = Vec::new();
    let mut pos = file_size;

    // Read chunks from the end until we have at least one full line we can
    // isolate (i.e., a `\n` preceded by more data or the start of file).
    loop {
        let chunk_size = pos.min(CHUNK);
        let start = pos - chunk_size;
        reader.seek(SeekFrom::Start(start))?;
        let mut buf = vec![0u8; chunk_size as usize];
        reader.read_exact(&mut buf)?;

        // Prepend: we're moving backwards.
        buf.extend_from_slice(&tail);
        tail = buf;
        pos = start;

        // Strip trailing newlines, then look for the preceding newline to
        // delimit the last complete line.
        let trimmed_end = tail.iter().rposition(|&b| b != b'\n' && b != b'\r');
        let content = match trimmed_end {
            Some(end) => &tail[..=end],
            None => {
                // Nothing but newlines in the whole read-so-far.
                if pos == 0 { return Ok(None); }
                continue;
            }
        };
        let newline_pos = content.iter().rposition(|&b| b == b'\n');
        let last_line_bytes = match (newline_pos, pos) {
            (Some(p), _) => &content[p + 1..],
            (None, 0) => content, // reached start of file, the whole thing is one line
            (None, _) => { continue; } // need to read more from earlier in the file
        };

        let line = std::str::from_utf8(last_line_bytes).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("non-utf8 in {}: {}", path.display(), e))
        })?;
        if line.is_empty() {
            return Ok(None);
        }
        let frame: FrameJsonl = serde_json::from_str(line).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("corrupt last frame in {}: {}", path.display(), e),
            )
        })?;
        return Ok(Some(frame.frame_id));
    }
}

/// Open or create a `.mv2d` file for appending.
pub fn open_mv2d(path: &Path) -> Result<File> {
    let file = OpenOptions::new()
        .create(true)
        .read(true)
        .append(true)
        .open(path)?;
    Ok(file)
}

fn now_unix() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use tempfile::NamedTempFile;

    #[test]
    fn roundtrip_single_frame() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let mut file = open_mv2d(&path).unwrap();

        let frame = FrameJsonl::new(0, "Hello, Memvid!")
            .with_uri("mv2://test/001")
            .with_title("Test Frame")
            .with_tag("sector", "episodic");

        let offset = write_frame_jsonl(&mut file, &frame).unwrap();
        assert_eq!(offset, 0);

        let frames = read_all_frames(&path).unwrap();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].1.frame_id, 0);
        assert_eq!(frames[0].1.content, "Hello, Memvid!");
        assert_eq!(frames[0].1.tags.get("sector").unwrap(), "episodic");
    }

    #[test]
    fn append_multiple_frames() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let mut file = open_mv2d(&path).unwrap();

        for i in 0..5 {
            let frame = FrameJsonl::new(i, format!("Frame {i}"));
            write_frame_jsonl(&mut file, &frame).unwrap();
        }

        let frames = read_all_frames(&path).unwrap();
        assert_eq!(frames.len(), 5);
        assert_eq!(frames[4].1.frame_id, 4);
        assert_eq!(frames[4].1.content, "Frame 4");
    }

    #[test]
    fn last_frame_id_empty_file() {
        let tmp = NamedTempFile::new().unwrap();
        assert_eq!(last_frame_id(tmp.path()).unwrap(), None);
    }

    #[test]
    fn last_frame_id_with_frames() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let mut file = open_mv2d(&path).unwrap();

        for i in 0..10 {
            let frame = FrameJsonl::new(i, format!("Frame {i}"));
            write_frame_jsonl(&mut file, &frame).unwrap();
        }

        assert_eq!(last_frame_id(&path).unwrap(), Some(9));
    }

    #[test]
    fn tombstone_frame() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let mut file = open_mv2d(&path).unwrap();

        let frame = FrameJsonl::new(0, "Original content");
        write_frame_jsonl(&mut file, &frame).unwrap();

        let tombstone = FrameJsonl::tombstone(1, 0);
        write_frame_jsonl(&mut file, &tombstone).unwrap();

        let frames = read_all_frames(&path).unwrap();
        assert_eq!(frames.len(), 2);
        assert!(frames[0].1.is_active());
        assert!(!frames[1].1.is_active());
        assert_eq!(frames[1].1.supersedes, Some(0));
    }

    #[test]
    fn cat_readable_output() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let mut file = open_mv2d(&path).unwrap();

        let frame = FrameJsonl::new(0, "SurrealDB droppt Connections nach 32min Idle")
            .with_tag("agent", "anubis")
            .with_tag("sector", "episodic");
        write_frame_jsonl(&mut file, &frame).unwrap();
        drop(file);

        // Verify the file is actually cat-readable JSONL
        let mut content = String::new();
        File::open(&path).unwrap().read_to_string(&mut content).unwrap();
        let lines: Vec<&str> = content.trim().split('\n').collect();
        assert_eq!(lines.len(), 1);
        assert!(lines[0].contains("SurrealDB"));
        assert!(lines[0].contains("anubis"));
    }

    #[test]
    fn supersedes_frame() {
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        let mut file = open_mv2d(&path).unwrap();

        let v1 = FrameJsonl::new(0, "Original system prompt");
        write_frame_jsonl(&mut file, &v1).unwrap();

        let v2 = FrameJsonl::new(1, "Updated system prompt")
            .with_supersedes(0);
        write_frame_jsonl(&mut file, &v2).unwrap();

        let frames = read_all_frames(&path).unwrap();
        assert_eq!(frames[1].1.supersedes, Some(0));
    }
}
